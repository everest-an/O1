import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module, config) -> None:
    """GPT-2-style init with MT-specific overrides."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    # MT-specific overrides applied by name after full model init
    # (called from MTLNNModel.__init__ via named_modules loop)


def init_mt_params(model: nn.Module, config) -> None:
    """Override specific MT parameter initialisations after standard init."""
    for name, param in model.named_parameters():
        if "log_tau" in name:
            # τ = softplus(log_tau) + tau_min ≈ tau_init when log_tau = log(exp(tau_init - tau_min) - 1)
            target = math.log(math.exp(config.tau_init - config.tau_min) - 1.0 + 1e-6)
            nn.init.constant_(param, target)
        elif "gamma" in name and "gtp" not in name.lower():
            nn.init.constant_(param, config.gamma_init)
        elif "polarity_direction" in name:
            nn.init.uniform_(param, -0.1, 0.1)
        elif "W_lat" in name:
            # Near-identity: eye + small noise
            nn.init.eye_(param)
            param.data += torch.randn_like(param) * 0.01
        elif "coherence_scale" in name:
            nn.init.constant_(param, 0.1)
        elif "collapse_threshold" in name:
            nn.init.constant_(param, 0.5)
        elif "blend_weights" in name:
            nn.init.zeros_(param)           # softmax(zeros) = uniform blend


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_module: Dict[str, int] = {}
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        by_module[name] = n
    return {"total": total, "trainable": trainable, "by_module": by_module}


# ---------------------------------------------------------------------------
# Causal mask helper
# ---------------------------------------------------------------------------

def get_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """(T, T) lower-triangular boolean mask; True = keep."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


# ---------------------------------------------------------------------------
# Learning-rate scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._step = 0
        # Store base LRs per param group
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self._step += 1
        s = self._step
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            if s < self.warmup_steps:
                lr = base_lr * s / max(1, self.warmup_steps)
            else:
                progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            pg["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str,
    config=None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if config is not None:
        import dataclasses
        payload["config"] = dataclasses.asdict(config)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ---------------------------------------------------------------------------
# AdamW param groups for MT-LNN (separate LRs for dynamical constants)
# ---------------------------------------------------------------------------

def make_param_groups(model: nn.Module, base_lr: float) -> list:
    """
    Separate param groups so ODE constants (τ, γ) train at 0.33× base_lr
    and polarity can move faster at 1.67× base_lr.
    """
    ode_names = {"log_tau", "gtp_gamma"}
    polarity_names = {"polarity_direction"}
    lateral_names = {"W_lat"}

    ode_params, polarity_params, lateral_params, main_params = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        base = name.split(".")[-1]
        if base in ode_names:
            ode_params.append(param)
        elif base in polarity_names:
            polarity_params.append(param)
        elif base in lateral_names:
            lateral_params.append(param)
        else:
            main_params.append(param)

    groups = [
        {"params": main_params,     "lr": base_lr,          "weight_decay": 0.1},
        {"params": ode_params,      "lr": base_lr * 0.33,   "weight_decay": 0.0},
        {"params": polarity_params, "lr": base_lr * 1.67,   "weight_decay": 0.0},
        {"params": lateral_params,  "lr": base_lr * 0.33,   "weight_decay": 0.01},
    ]
    # Drop empty groups
    return [g for g in groups if len(g["params"]) > 0]
