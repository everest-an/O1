import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn.llama_adapter import attach_mt_adapters, count_trainable_parameters


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        return (self.proj(hidden_states),)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size=32, n_layers=4):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [TinyDecoderLayer(hidden_size) for _ in range(n_layers)]
        )

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def test_attach_mt_adapters_freezes_base_and_trains_adapter():
    torch.manual_seed(0)
    model = TinyBackbone()
    wrapped = attach_mt_adapters(
        model,
        every=2,
        n_protofilaments=4,
        n_time_scales=2,
        map_hidden_dim=8,
        init_scale=1e-2,
    )
    assert wrapped == [1, 3]
    assert count_trainable_parameters(model) > 0
    assert all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if ".base_layer." in name
    )

    x = torch.randn(2, 6, 32)
    out = model(x)
    assert out.shape == x.shape
    out.square().mean().backward()

    adapter_grads = [
        p.grad
        for name, p in model.named_parameters()
        if "mt_adapter" in name and p.requires_grad
    ]
    assert adapter_grads and all(g is not None for g in adapter_grads)
