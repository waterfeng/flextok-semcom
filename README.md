# FlexTok: Resampling Images into 1D Token Sequences of Flexible Length

[`Website`](https://flextok.epfl.ch) | [`arXiv`](https://arxiv.org/abs/2502.13967) | [`🤗 Demo`](https://huggingface.co/spaces/EPFL-VILAB/FlexTok) | [`BibTeX`](#citation)

Official implementation and pre-trained models for: <br>
[**FlexTok: Resampling Images into 1D Token Sequences of Flexible Length**](https://arxiv.org/abs/2502.13967), ICML 2025 <br>
*[Roman Bachmann](https://roman-bachmann.github.io/)\*, [Jesse Allardice](https://github.com/JesseAllardice)\*, [David Mizrahi](https://dmizrahi.com/)\*, [Enrico Fini](https://scholar.google.com/citations?user=OQMtSKIAAAAJ), [Oğuzhan Fatih Kar](https://ofkar.github.io/), [Elmira Amirloo](https://elamirloo.github.io/), [Alaaeldin El-Nouby](https://aelnouby.github.io/), [Amir Zamir](https://vilab.epfl.ch/zamir/), [Afshin Dehghan](https://scholar.google.com/citations?user=wcX-UW4AAAAJ)*


![FlexTok main figure](./assets/flextok_pull_darkmode.png#gh-dark-mode-only)
![FlexTok main figure](./assets/flextok_pull_lightmode.png#gh-light-mode-only)


## Table of contents
- [Usage](#usage)
    - [Installation](#installation)
    - [Getting started](#getting-started)
- [Model Zoo](#model-zoo)
    - [FlexTok tokenizers](#flextok-tokenizers)
    - [VAEs](#vaes)
- [License](#license)
- [Citation](#citation)


## Usage

### Installation
1. Clone this repository and navigate to the root directory:
```bash
git clone https://github.com/waterfeng/flextok-semcom.git
cd flextok-semcom
```

2. Create a new conda environment, then install the dependencies (Python >=3.10):
```bash
conda create -n flextok python=3.10 -y
conda activate flextok
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run Python from the repository root to import the local `flextok` source. For scripts in subdirectories, use module execution from the root, for example `python -m finetune.train` for `finetune/train.py`.

3. Verify that CUDA is available in PyTorch by running the following in a Python shell:
```bash
# Run in Python shell
import torch
print(torch.cuda.is_available())  # Should return True
```
If CUDA is not available, consider re-installing PyTorch following the [official installation instructions](https://pytorch.org/get-started/locally/).

4. (Optional) Expose the new conda environment as a kernel to Jupyter notebooks:
```bash
pip install ipykernel
python -m ipykernel install --user --name flextok --display-name "FlexTok (flextok)"
```


### Getting started
We recommend checking out the Jupyter notebook in [notebooks/flextok_inference.ipynb](notebooks/flextok_inference.ipynb) to get started with FlexTok tokenizer and VAE inference.
Please see the [Model Zoo](#model-zoo) for all available FlexTok and VAE models, as well as sample code snippets that illustrate encoding and decoding.


## Model Zoo
We provide FlexTok and VAE checkpoints as [safetensors](https://huggingface.co/docs/safetensors/en/index), and also offer easy loading via [Hugging Face Hub](https://huggingface.co/docs/hub/index). 

Note that we instantiate the models from configs stored in the safetensors' metadata using `hydra.utils.instantiate`, which may be vulnerable to arbitrary code execution. While we take steps to sanitize the loaded metadata before passing it to `hydra.utils.instantiate`, we recommend only using FlexTok checkpoints from trusted sources and inspecting the metadata manually when loading unofficial checkpoints.

### FlexTok tokenizers

| Encoder layers | Decoder layers | Dataset | HF Hub | Safetensors |
| -------------- | -------------- | ------- | ------ | ----------- |
| 12 | 12 | IN1K | [EPFL-VILAB/flextok_d12_d12_in1k](https://huggingface.co/EPFL-VILAB/flextok_d12_d12_in1k) | [Checkpoint](https://huggingface.co/EPFL-VILAB/flextok_d12_d12_in1k/resolve/main/model.safetensors) |
| 18 | 18 | IN1K | [EPFL-VILAB/flextok_d18_d18_in1k](https://huggingface.co/EPFL-VILAB/flextok_d18_d18_in1k) | [Checkpoint](https://huggingface.co/EPFL-VILAB/flextok_d18_d18_in1k/resolve/main/model.safetensors) |
| 18 | 28 | IN1K | [EPFL-VILAB/flextok_d18_d28_in1k](https://huggingface.co/EPFL-VILAB/flextok_d18_d28_in1k) | [Checkpoint](https://huggingface.co/EPFL-VILAB/flextok_d18_d28_in1k/resolve/main/model.safetensors) |
| 18 | 28 | DFN  | [EPFL-VILAB/flextok_d18_d28_dfn](https://huggingface.co/EPFL-VILAB/flextok_d18_d28_dfn) | [Checkpoint](https://huggingface.co/EPFL-VILAB/flextok_d18_d28_dfn/resolve/main/model.safetensors) |

Example usage, loading a `FlexTok d18-d28 DFN` model directly from HuggingFace Hub:
```python
from flextok.flextok_wrapper import FlexTokFromHub
model = FlexTokFromHub.from_pretrained('EPFL-VILAB/flextok_d18_d28_dfn').eval()
```

The model can also be loaded by downloading the safetensors checkpoint manually and loading it using our helper functions:
```python
from hydra.utils import instantiate
from flextok.utils.checkpoint import load_safetensors

ckpt, config = load_safetensors('/path/to/model.safetensors')
model = instantiate(config).eval()
model.load_state_dict(ckpt)
```

After loading a FlexTok model, image batches can be encoded using:
```python
from flextok.utils.demo import imgs_from_urls
# Load example images of shape (B, 3, 256, 256), normalized to [-1,1]
imgs = imgs_from_urls(urls=['https://storage.googleapis.com/flextok_site/nb_demo_images/0.png'])

# tokens_list is a list of [1, 256] discrete token sequences
tokens_list = model.tokenize(imgs)
```

The list of token sequences can be truncated in a nested fashion:
```python
k_keep = 64 # For example, only keep the first 64 out of 256 tokens
tokens_list = [t[:,:k_keep] for t in tokens_list]
```

To decode the tokens with FlexTok's rectified flow decoder, call:
```python
# tokens_list is a list of [1, l] discrete token sequences, with l <= 256
# reconst is a [B, 3, 256, 256] tensor, normalized to [-1,1]
reconst = model.detokenize(
    tokens_list,
    timesteps=20, # Number of denoising steps
    guidance_scale=7.5, # Classifier-free guidance scale
    perform_norm_guidance=True, # See https://arxiv.org/abs/2410.02416
)
```

### VAEs

| Latent channels | Downsampling factor | HF Hub | Safetensors |
| --------------- | ------------------- | ------ | ----------- |
| 4  | 8 | [EPFL-VILAB/flextok_vae_c4](https://huggingface.co/EPFL-VILAB/flextok_vae_c4) | [Checkpoint](https://huggingface.co/EPFL-VILAB/flextok_vae_c4/resolve/main/diffusion_pytorch_model.safetensors) |
| 8  | 8 | [EPFL-VILAB/flextok_vae_c8](https://huggingface.co/EPFL-VILAB/flextok_vae_c8) | [Checkpoint](https://huggingface.co/EPFL-VILAB/flextok_vae_c8/resolve/main/diffusion_pytorch_model.safetensors) |
| 16 | 8 | [EPFL-VILAB/flextok_vae_c16](https://huggingface.co/EPFL-VILAB/flextok_vae_c16) | [Checkpoint](https://huggingface.co/EPFL-VILAB/flextok_vae_c16/resolve/main/diffusion_pytorch_model.safetensors) |


Example usage, loading an `AutoencoderKL` directly from HuggingFace Hub and autoencoding a sample image:
```python
from diffusers.models import AutoencoderKL
from flextok.utils.demo import imgs_from_urls

vae = AutoencoderKL.from_pretrained(
    'EPFL-VILAB/flextok_vae_c16', low_cpu_mem_usage=False
).eval()

# Load image of shape (B, 3, H, W), normalized to [-1,1]
imgs = imgs_from_urls(urls=['https://storage.googleapis.com/four_m_site/images/demo_rgb.png'])

# Autoencode with the VAE
latents = vae.encode(imgs).latent_dist.sample() # Shape (B, D, H//8, W//8) with D in 4, 8, 16
reconst = vae.decode(latents).sample # Shape (B, 3, H, W)
```


## License
The code in this repository is released under the license as found in the [LICENSE](LICENSE) file.

The model weights in this repository are released under the Apple Machine Learning Research Model license as found in the [LICENSE_WEIGHTS](LICENSE_WEIGHTS) file.


## Citation
If you find this repository helpful, please consider citing our work:
```
@article{flextok,
    title={{FlexTok}: Resampling Images into 1D Token Sequences of Flexible Length},
    author={Roman Bachmann and Jesse Allardice and David Mizrahi and Enrico Fini and O{\u{g}}uzhan Fatih Kar and Elmira Amirloo and Alaaeldin El-Nouby and Amir Zamir and Afshin Dehghan},
    journal={arXiv 2025},
    year={2025},
}
```
