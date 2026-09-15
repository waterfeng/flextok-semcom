#!/usr/bin/env python3
"""
Fine-tune FlexTok on a custom image dataset.

This script is designed to run inside the official FlexTok repository:
    https://github.com/apple-aiml-research/ml-flextok

Important:
- The official repository currently exposes model/inference code and pretrained
  checkpoints, but not the original training loop.
- Therefore, this script implements a practical fine-tuning loop directly from
  the released FlexTok forward/noise/pipeline definitions.
- The rectified-flow target is:
      v* = eps - x0
  because the released noise module constructs
      x_t = sigma * eps + (1 - sigma) * x0
  and the released inference pipeline integrates with
      x_next = x - dt * v_theta(x_t, t, z).

Expected dataset layout:
    data_root/
        class_or_folder_a/
            0001.jpg
            0002.png
        class_or_folder_b/
            ...
or simply any nested tree of image files. Labels are ignored.

Example:
    python -m  finetune.train_flextok_finetune \
        --data_root /path/to/satellite_images \
        --model_id EPFL-VILAB/flextok_d12_d12_in1k \
        --output_dir outputs/flextok_satellite \
        --image_size 256 \
        --batch_size 4 \
        --grad_accum_steps 4 \
        --epochs 20 \
        --lr 1e-5 \
        --precision bf16

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=4 -m  finetune.train_flextok_finetune \
        --data_root /path/to/images \
        --output_dir outputs/flextok_satellite \
        --batch_size 4 \
        --precision bf16

Dependencies in addition to the official FlexTok environment:
    pip install torchvision pillow
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from flextok.flextok_wrapper import FlexTokFromHub


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")


class RecursiveImageDataset(Dataset):
    """Recursively load images; directory names/classes are ignored."""

    def __init__(self, root: str | Path, image_size: int = 256, train: bool = True):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.files = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No supported image files found under: {self.root}")

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # FlexTok README expects RGB images normalized to [-1, 1].
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    image_size,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
                return self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {path}") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("FlexTok custom-data fine-tuning")

    # Data/model
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument(
        "--model_id",
        type=str,
        default="EPFL-VILAB/flextok_d12_d12_in1k",
        help="Hugging Face FlexTok checkpoint.",
    )
    p.add_argument("--output_dir", type=str, default="outputs/flextok_finetune")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)

    # Optimization
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Which parts to fine-tune
    p.add_argument(
        "--train_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fine-tune the FlexTok encoder.",
    )
    p.add_argument(
        "--train_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fine-tune the rectified-flow decoder.",
    )
    p.add_argument(
        "--train_regularizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fine-tune learnable regularizer parameters, if present.",
    )

    # Precision/runtime
    p.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument(
        "--val_keep_k",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256],
        help="Fixed token lengths used for validation.",
    )

    # Resume
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


def distributed_info() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        local_rank, rank = 0, 0

    return distributed, local_rank, rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def configure_trainable_parts(model: torch.nn.Module, args: argparse.Namespace) -> None:
    # VAE is intentionally kept frozen. FlexTok.encode() already invokes its VAE
    # inside torch.no_grad(), matching the released model design.
    for p in model.vae.parameters():
        p.requires_grad = False
    model.vae.eval()

    for p in model.encoder.parameters():
        p.requires_grad = args.train_encoder

    for p in model.decoder.parameters():
        p.requires_grad = args.train_decoder

    for p in model.regularizer.parameters():
        p.requires_grad = args.train_regularizer

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if n_train == 0:
        raise ValueError("No trainable parameters. Enable encoder/decoder/regularizer.")

    print(f"Trainable parameters: {n_train / 1e6:.2f}M / {n_total / 1e6:.2f}M")


def build_datasets(
    root: str,
    image_size: int,
    val_ratio: float,
    seed: int,
):
    # Build one list first so train and validation split reference the same files.
    full = RecursiveImageDataset(root, image_size=image_size, train=True)
    n = len(full)
    n_val = int(round(n * val_ratio))
    n_val = max(1, n_val) if n > 1 and val_ratio > 0 else 0
    n_val = min(n_val, max(0, n - 1))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(n),
        [n_train, n_val],
        generator=generator,
    )

    train_ds = RecursiveImageDataset(root, image_size=image_size, train=True)
    train_ds.files = [train_ds.files[i] for i in train_subset.indices]

    val_ds = None
    if n_val > 0:
        val_ds = RecursiveImageDataset(root, image_size=image_size, train=False)
        val_ds.files = [val_ds.files[i] for i in val_subset.indices]

    return train_ds, val_ds


def to_data_dict(model: torch.nn.Module, images: torch.Tensor) -> dict:
    """
    FlexTok expects a list of per-sample tensors [1,C,H,W].
    """
    return {model.vae.images_read_key: list(images.split(1, dim=0))}


def _mean_per_sample_mse(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
) -> torch.Tensor:
    if len(predictions) != len(targets):
        raise RuntimeError(
            f"Prediction/target batch mismatch: {len(predictions)} vs {len(targets)}"
        )

    losses = [
        F.mse_loss(pred.float(), target.float(), reduction="mean")
        for pred, target in zip(predictions, targets)
    ]
    return torch.stack(losses).mean()


def compute_rectified_flow_loss(data_dict: dict) -> torch.Tensor:
    """
    Loss implied by the released FlexTok min-RF code.

    Released noise process:
        x_t = sigma * eps + (1 - sigma) * x0

    Therefore:
        dx_t / d sigma = eps - x0

    The decoder output stored in `vae_latents_reconst` is used as velocity
    by the released inference integrator:
        x_next = x - dt * model_output

    Thus the velocity regression target is eps - x0.
    """
    pred_velocity = data_dict["vae_latents_reconst"]
    noise = data_dict["flow_noise"]
    clean = data_dict["vae_latents"]

    target_velocity = [
        eps - x0
        for eps, x0 in zip(noise, clean)
    ]

    return _mean_per_sample_mse(pred_velocity, target_velocity)


def forward_train(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    data_dict = to_data_dict(model, images)
    out = model(data_dict)
    return compute_rectified_flow_loss(out)


@torch.no_grad()
def forward_eval_fixed_k(
    model: torch.nn.Module,
    images: torch.Tensor,
    keep_k: int,
) -> torch.Tensor:
    """
    Validate the RF objective at a deterministic token length.

    In eval mode, MaskedNestedDropout keeps all tokens unless its
    `eval_keep_k` key is supplied. The released model exposes that key
    through decoder.module_dict["dec_nested_dropout"].
    """
    data_dict = to_data_dict(model, images)
    data_dict = model.encode(data_dict)

    nested_dropout = model.decoder.module_dict["dec_nested_dropout"]
    eval_key = nested_dropout.eval_keep_k_read_key
    if eval_key is None:
        raise RuntimeError("Decoder nested-dropout module has no eval_keep_k key.")

    batch_size = images.shape[0]
    max_k = model.encoder.module_dict["enc_register_module"].n_max
    if not (1 <= keep_k <= max_k):
        raise ValueError(f"keep_k={keep_k} outside valid range [1, {max_k}]")

    data_dict[eval_key] = [keep_k] * batch_size
    data_dict = model.flow_matching_noise_module(data_dict)
    data_dict = model.decoder(data_dict)

    return compute_rectified_flow_loss(data_dict)


def cosine_lr_lambda(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0

    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, step / warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()

    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def all_reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    state: TrainState,
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)

    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": asdict(state),
        "args": vars(args),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
) -> TrainState:
    ckpt = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return TrainState(**ckpt.get("state", {}))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    precision: str,
    keep_ks: Iterable[int],
    distributed: bool,
) -> dict:
    if loader is None:
        return {}

    raw_model = unwrap_model(model)
    raw_model.eval()

    metrics = {}
    for k in keep_ks:
        total = torch.zeros((), device=device)
        count = torch.zeros((), device=device)

        for images in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, precision):
                loss = forward_eval_fixed_k(raw_model, images, keep_k=k)

            total += loss.detach() * images.shape[0]
            count += images.shape[0]

        if distributed:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        metrics[f"val_rf_loss_k{k}"] = (total / count.clamp_min(1)).item()

    # Restore normal train mode; frozen VAE stays in eval due its wrapper behavior.
    raw_model.train()
    raw_model.vae.eval()
    return metrics


def main():
    args = parse_args()
    distributed, local_rank, rank, world_size = distributed_info()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This script is intended for CUDA training. "
            "The official FlexTok setup also assumes CUDA."
        )

    device = torch.device("cuda", local_rank)
    seed_everything(args.seed, rank)

    out_dir = Path(args.output_dir)
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if is_main_process(rank):
        print(f"Loading pretrained FlexTok: {args.model_id}")

    model = FlexTokFromHub.from_pretrained(args.model_id)
    configure_trainable_parts(model, args)
    model.to(device)

    if args.compile:
        # Compile only after moving to the target device.
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)

    train_ds, val_ds = build_datasets(
        args.data_root,
        args.image_size,
        args.val_ratio,
        args.seed,
    )

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, drop_last=False)
        if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False, drop_last=False)
        if distributed and val_ds is not None else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )

    if is_main_process(rank):
        print(
            f"Dataset: train={len(train_ds)}, "
            f"val={0 if val_ds is None else len(val_ds)}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = args.epochs * updates_per_epoch
    warmup_steps = int(total_updates * args.warmup_ratio)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(
            step,
            total_steps=total_updates,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        ),
    )

    scaler = None
    if args.precision == "fp16":
        scaler = torch.amp.GradScaler("cuda")

    state = TrainState()
    if args.resume is not None:
        state = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        if is_main_process(rank):
            print(
                f"Resumed from {args.resume}: "
                f"epoch={state.epoch}, global_step={state.global_step}"
            )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(state.epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        raw_model = unwrap_model(model)
        raw_model.train()
        raw_model.vae.eval()

        epoch_loss = 0.0
        epoch_items = 0
        start_time = time.time()

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            # Prevent unnecessary DDP gradient synchronization on intermediate
            # gradient-accumulation microsteps.
            is_update_step = (
                ((batch_idx + 1) % args.grad_accum_steps == 0)
                or (batch_idx + 1 == len(train_loader))
            )

            sync_context = nullcontext()
            if isinstance(model, DDP) and not is_update_step:
                sync_context = model.no_sync()

            with sync_context:
                with autocast_context(device, args.precision):
                    loss = forward_train(model, images)
                    loss_for_backward = loss / args.grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            epoch_loss += loss.detach().float().item() * batch_size
            epoch_items += batch_size

            if is_update_step:
                if args.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        args.max_grad_norm,
                    )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state.global_step += 1

                if (
                    is_main_process(rank)
                    and state.global_step % args.log_every == 0
                ):
                    reduced_loss = all_reduce_mean(
                        loss.detach().float(), distributed
                    )
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch + 1:03d}/{args.epochs:03d} "
                        f"step={state.global_step:07d}/{total_updates:07d} "
                        f"rf_loss={reduced_loss.item():.6f} "
                        f"lr={lr:.3e}"
                    )

        # Aggregate epoch train loss.
        stats = torch.tensor(
            [epoch_loss, epoch_items],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        mean_train_loss = (stats[0] / stats[1].clamp_min(1)).item()

        val_metrics = validate(
            model,
            val_loader,
            device,
            args.precision,
            args.val_keep_k,
            distributed,
        )

        if is_main_process(rank):
            elapsed = time.time() - start_time
            val_text = " ".join(
                f"{key}={value:.6f}"
                for key, value in val_metrics.items()
            )
            print(
                f"[epoch {epoch + 1}] "
                f"train_rf_loss={mean_train_loss:.6f} "
                f"{val_text} time={elapsed:.1f}s"
            )

            state.epoch = epoch + 1

            # Use the longest-token validation objective for "best" checkpoint
            # when present, otherwise use average validation objective.
            if val_metrics:
                preferred_key = f"val_rf_loss_k{max(args.val_keep_k)}"
                score = val_metrics.get(
                    preferred_key,
                    sum(val_metrics.values()) / len(val_metrics),
                )
            else:
                score = mean_train_loss

            if score < state.best_val_loss:
                state.best_val_loss = score
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    state,
                    args,
                )

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    out_dir / f"epoch_{epoch + 1:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    state,
                    args,
                )

            save_checkpoint(
                out_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                state,
                args,
            )

        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
