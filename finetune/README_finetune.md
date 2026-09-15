# FlexTok fine-tuning helper

This folder contains a custom fine-tuning entry point for the released
Apple/EPFL FlexTok implementation.

## 1. Environment

Clone and install the official repository first:

```bash
git clone https://github.com/apple-aiml-research/ml-flextok
cd ml-flextok
conda create -n flextok python=3.10 -y
conda activate flextok
pip install -e .
pip install torchvision pillow pytest
```

Copy `train_flextok_finetune.py` into the repository root.

## 2. Dataset

Any recursively nested directory of common image formats is accepted:

```text
satellite_data/
├── scene_001.png
├── scene_002.jpg
└── subfolder/
    ├── scene_003.tif
    └── scene_004.png
```

Class labels are not needed.

## 3. Single-GPU training

```bash
python -m finetune.train_flextok_finetune \
  --data_root ~/Documents/b412/zhaoxinfeng/flextok-semcom/datasets/UCMLUD \
  --model_id ~/Documents/b412/zhaoxinfeng/flextok-semcom/models/flextok_d18_d28_dfn \
  --output_dir outputs/satellite_flextok \
  --image_size 256 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --epochs 20 \
  --lr 1e-5 \
  --precision bf16
```

## 4. Multi-GPU training

```bash
torchrun --standalone --nproc_per_node=4 train_flextok_finetune.py \
  --data_root /path/to/satellite_data \
  --output_dir outputs/satellite_flextok \
  --batch_size 4 \
  --grad_accum_steps 2 \
  --precision bf16
```

## 5. Resume

```bash
python train_flextok_finetune.py \
  --data_root /path/to/satellite_data \
  --output_dir outputs/satellite_flextok \
  --resume outputs/satellite_flextok/last.pt
```

## 6. Fine-tuning subsets

Decoder only:

```bash
python train_flextok_finetune.py \
  --data_root /path/to/data \
  --no-train_encoder \
  --train_decoder
```

Encoder + decoder (default):

```bash
python train_flextok_finetune.py \
  --data_root /path/to/data \
  --train_encoder \
  --train_decoder
```

## 7. Validation token lengths

The released IN1K checkpoint trains nested dropout over powers of two.
For validation this script explicitly evaluates fixed lengths:

```bash
--val_keep_k 1 4 16 64 256
```

For the satellite semantic-communication project, a useful next step is
to replace or augment this with task-specific semantic utility at each K,
for example classification accuracy / CLIP similarity / reconstruction
quality combined with communication latency or required channel uses.

## Important note

The official public FlexTok repository does not include its original training
loop. This helper reconstructs the rectified-flow fine-tuning objective from
the released `MinRFNoiseModule`, `FlexTok.forward`, and `MinRFPipeline`.
It should therefore be described as a *FlexTok fine-tuning implementation
based on the released model code*, not as Apple's official training script.
