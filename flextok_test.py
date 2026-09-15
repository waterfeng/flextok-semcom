#!/usr/bin/env python3
"""
FlexTok inference test script.

Features
--------
1. Loads the local FlexTok d18-d28 DFN safetensors checkpoint.
2. Supports either local input images or the official FlexTok demo images.
3. Encodes each image once into 256 discrete tokens.
4. Reconstructs images with flexible token lengths:
   1, 2, 4, 8, 16, 32, 64, 128, 256.
5. Saves all results under ./outputs/.

Run examples
------------
# Use official demo images
python flextok_test.py

# Use one or more local images
python flextok_test.py --input image1.jpg image2.png

# Only test selected token lengths
python flextok_test.py --input image.jpg --tokens 1 4 16 64 256

# Change denoising steps
python flextok_test.py --input image.jpg --timesteps 25
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from hydra.utils import instantiate

from flextok.utils.checkpoint import load_safetensors
from flextok.utils.demo import imgs_from_urls, denormalize
from flextok.utils.misc import detect_bf16_support, get_bf16_context, get_generator


DEFAULT_MODEL_PATH = (
    "~/Documents/b412/zhaoxinfeng/flextok-semcom/"
    "models/flextok_d18_d28_dfn_model.safetensors"
)

DEFAULT_DEMO_URLS = [
    f"https://storage.googleapis.com/flextok_site/nb_demo_images/{i}.png"
    for i in range(6)
]

DEFAULT_TOKEN_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def parse_args():
    parser = argparse.ArgumentParser(description="Test FlexTok flexible-length image tokenization.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local FlexTok .safetensors checkpoint.",
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help=(
            "Local image path(s). If omitted, the six official FlexTok demo "
            "images are downloaded and used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory used to save all outputs. Default: outputs",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=DEFAULT_TOKEN_LENGTHS,
        help="Token lengths used for reconstruction.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=25,
        help="Rectified-flow denoising steps. Default: 25",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale. Default: 7.5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by the decoder. Default: 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cuda, cuda:0, cpu. Default: auto",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show the decoder denoising progress bar.",
    )
    return parser.parse_args()


def configure_runtime(device_arg=None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    if device_arg is not None:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enable_bf16 = device.type == "cuda" and detect_bf16_support()

    print(f"[INFO] Device: {device}")
    print(f"[INFO] BF16 enabled: {enable_bf16}")
    return device, enable_bf16


def load_flextok_model(model_path, device):
    model_path = Path(model_path).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"FlexTok checkpoint not found:\n{model_path}\n"
            "Please check --model."
        )

    print(f"[INFO] Loading checkpoint:\n       {model_path}")
    ckpt, config = load_safetensors(str(model_path))
    model = instantiate(config).eval()
    model.load_state_dict(ckpt)
    model = model.to(device)

    print("[INFO] FlexTok model loaded successfully.")
    return model, model_path


def load_local_images(paths, device, image_size=256):
    """
    Load local RGB images and convert them to the same basic format expected
    by FlexTok: B x 3 x 256 x 256, float32, normalized to [-1, 1].

    Images are resized while preserving aspect ratio, then center-cropped.
    """
    tensors = []
    names = []

    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")

        img = Image.open(path).convert("RGB")

        width, height = img.size
        scale = image_size / min(width, height)
        new_width = max(image_size, round(width * scale))
        new_height = max(image_size, round(height * scale))

        img = TF.resize(
            img,
            [new_height, new_width],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        img = TF.center_crop(img, [image_size, image_size])

        tensor = TF.to_tensor(img)          # [0, 1]
        tensor = tensor * 2.0 - 1.0         # [-1, 1]

        tensors.append(tensor)
        names.append(path.stem)

    batch = torch.stack(tensors, dim=0).to(device)
    return batch, names


def load_images(input_paths, device):
    if input_paths:
        imgs, names = load_local_images(input_paths, device)
        print(f"[INFO] Loaded {len(names)} local image(s).")
        return imgs, names

    print("[INFO] No --input supplied. Loading official FlexTok demo images.")
    imgs = imgs_from_urls(DEFAULT_DEMO_URLS).to(device)
    names = [f"demo_{i}" for i in range(len(DEFAULT_DEMO_URLS))]
    return imgs, names


def tensor_to_pil(tensor):
    tensor = denormalize(tensor.unsqueeze(0))[0].clamp(0, 1).cpu()
    return TF.to_pil_image(tensor)


def save_ground_truth(imgs, names, output_dir):
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(names):
        tensor_to_pil(imgs[i]).save(gt_dir / f"{name}.png")


def save_tokens(tokens_list, names, output_dir):
    token_dir = output_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}

    for i, (tokens, name) in enumerate(zip(tokens_list, names)):
        tokens_cpu = tokens.detach().cpu()

        # Exact PyTorch representation.
        torch.save(tokens_cpu, token_dir / f"{name}_tokens.pt")

        # Human-readable JSON representation.
        values = tokens_cpu.squeeze(0).tolist()
        serializable[name] = values

        with open(token_dir / f"{name}_tokens.json", "w", encoding="utf-8") as f:
            json.dump(values, f)

    with open(token_dir / "all_tokens.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"[INFO] Saved token sequences to: {token_dir}")


def reconstruct_at_lengths(
    model,
    tokens_list,
    token_lengths,
    device,
    enable_bf16,
    timesteps,
    guidance_scale,
    seed,
    verbose,
):
    all_reconst = {}

    for k_keep in token_lengths:
        if not 1 <= k_keep <= 256:
            raise ValueError(f"Invalid token length {k_keep}; expected 1..256.")

        print(f"[INFO] Reconstructing with {k_keep:3d} token(s)...")

        subseq_list = [seq[:, :k_keep].clone() for seq in tokens_list]

        with get_bf16_context(enable_bf16):
            generator = get_generator(seed=seed, device=device)
            reconst = model.detokenize(
                subseq_list,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                perform_norm_guidance=True,
                generator=generator,
                verbose=verbose,
            )

        all_reconst[k_keep] = reconst.detach().cpu()

    return all_reconst


def save_reconstructions(all_reconst, names, output_dir):
    recon_root = output_dir / "reconstructions"
    recon_root.mkdir(parents=True, exist_ok=True)

    for k_keep, batch in all_reconst.items():
        token_dir = recon_root / f"{k_keep:03d}_tokens"
        token_dir.mkdir(parents=True, exist_ok=True)

        batch_plot = denormalize(batch).clamp(0, 1)

        for i, name in enumerate(names):
            image = TF.to_pil_image(batch_plot[i])
            image.save(token_dir / f"{name}.png")

    print(f"[INFO] Saved reconstructions to: {recon_root}")


def save_comparison_grid(imgs, all_reconst, token_lengths, names, output_dir, figscale=2.7):
    nrows = imgs.shape[0]
    ncols = len(token_lengths) + 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figscale * ncols, figscale * nrows),
        squeeze=False,
    )

    imgs_plot = denormalize(imgs.detach().cpu()).clamp(0, 1)
    reconst_plot = {
        k: denormalize(v).clamp(0, 1)
        for k, v in all_reconst.items()
    }

    for row in range(nrows):
        for col, k_keep in enumerate(token_lengths):
            axes[row, col].imshow(TF.to_pil_image(reconst_plot[k_keep][row]))
            axes[row, col].axis("off")

        axes[row, -1].imshow(TF.to_pil_image(imgs_plot[row]))
        axes[row, -1].axis("off")

        axes[row, 0].set_ylabel(names[row], fontsize=10)

    for col, k_keep in enumerate(token_lengths):
        suffix = "token" if k_keep == 1 else "tokens"
        axes[0, col].set_title(f"{k_keep} {suffix}", fontsize=12)

    axes[0, -1].set_title("Ground Truth", fontsize=12)

    plt.tight_layout()
    grid_path = output_dir / "flexible_length_comparison.png"
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved comparison grid to: {grid_path}")


def save_run_config(args, model_path, output_dir, device, enable_bf16):
    config = {
        "model": str(model_path),
        "input": args.input,
        "token_lengths": args.tokens,
        "timesteps": args.timesteps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "device": str(device),
        "bf16": bool(enable_bf16),
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()

    # Remove duplicates while retaining command-line order.
    token_lengths = list(dict.fromkeys(args.tokens))
    args.tokens = token_lengths

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device, enable_bf16 = configure_runtime(args.device)
    model, model_path = load_flextok_model(args.model, device)
    imgs, names = load_images(args.input, device)

    print("[INFO] Tokenizing images once into 256-token sequences...")
    with get_bf16_context(enable_bf16):
        tokens_list = model.tokenize(imgs)

    for name, tokens in zip(names, tokens_list):
        print(
            f"[INFO] {name}: token tensor shape = {tuple(tokens.shape)}, "
            f"dtype = {tokens.dtype}"
        )

    save_ground_truth(imgs, names, output_dir)
    save_tokens(tokens_list, names, output_dir)

    all_reconst = reconstruct_at_lengths(
        model=model,
        tokens_list=tokens_list,
        token_lengths=token_lengths,
        device=device,
        enable_bf16=enable_bf16,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        verbose=args.verbose,
    )

    save_reconstructions(all_reconst, names, output_dir)
    save_comparison_grid(
        imgs=imgs,
        all_reconst=all_reconst,
        token_lengths=token_lengths,
        names=names,
        output_dir=output_dir,
    )
    save_run_config(args, model_path, output_dir, device, enable_bf16)

    print("\n[DONE] FlexTok test finished.")
    print(f"[DONE] All outputs are under:\n       {output_dir}")


if __name__ == "__main__":
    main()
