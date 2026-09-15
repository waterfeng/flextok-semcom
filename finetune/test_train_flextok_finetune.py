"""
Small unit tests for train_flextok_finetune.py.

These tests intentionally avoid loading the large pretrained FlexTok checkpoint.
Run:
    pytest -q test_train_flextok_finetune.py
"""

import torch

from train_flextok_finetune import (
    _mean_per_sample_mse,
    compute_rectified_flow_loss,
    cosine_lr_lambda,
)


def test_per_sample_mse_zero_for_equal_tensors():
    xs = [torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)]
    loss = _mean_per_sample_mse(xs, [x.clone() for x in xs])
    assert torch.allclose(loss, torch.tensor(0.0))


def test_rectified_flow_target_is_noise_minus_clean():
    clean = [torch.zeros(1, 2, 2, 2)]
    noise = [torch.ones(1, 2, 2, 2)]
    exact_velocity = [torch.ones(1, 2, 2, 2)]

    data = {
        "vae_latents": clean,
        "flow_noise": noise,
        "vae_latents_reconst": exact_velocity,
    }
    loss = compute_rectified_flow_loss(data)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_cosine_schedule_starts_with_warmup_and_ends_at_floor():
    assert cosine_lr_lambda(0, 100, 10, 0.1) < 1e-6
    assert abs(cosine_lr_lambda(10, 100, 10, 0.1) - 1.0) < 1e-6
    assert abs(cosine_lr_lambda(100, 100, 10, 0.1) - 0.1) < 1e-6
