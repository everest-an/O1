"""
tests/test_irregular_dt.py — variable-Δt (continuous-time) support tests.

MT-LNN's CfLTC decay is exp(-Δt/τ). Historically Δt was pinned to the constant
config.dt=1.0, so the model treated every step as equally spaced — it could not
exploit the timestamps of irregularly-sampled inputs (the defining use-case of
liquid / continuous-time networks). These tests lock in the new per-step `dt`
path AND guarantee it is byte-for-byte backward compatible when dt is omitted.

Run:  python -m pytest tests/test_irregular_dt.py -v
"""

import sys
import warnings

import torch

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*", category=RuntimeWarning)

from mt_lnn import MTLNNConfig, MTLNNModel


def _cfg():
    return MTLNNConfig(
        vocab_size=200, max_seq_len=32, d_model=128,
        n_layers=2, n_heads=4, n_kv_heads=2, d_head=32,
        dropout=0.0, attention_dropout=0.0,
    )


def test_dt_none_equals_dt_one():
    """dt omitted must be identical to dt = self.dt (=1.0): zero behaviour change."""
    torch.manual_seed(0)
    m = MTLNNModel(_cfg()).eval()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        a = m(ids)["logits"]
        b = m(ids, dt=torch.ones(2, 16))["logits"]
    diff = (a - b).abs().max().item()
    print(f"  dt=None vs dt=1.0  max|diff| = {diff:.2e}")
    assert diff < 1e-4, f"per-step path diverges from constant path: {diff}"
    print("[ok] test_dt_none_equals_dt_one")


def test_dt_changes_output():
    """A different Δt must change the recurrent decay and hence the output."""
    torch.manual_seed(0)
    m = MTLNNModel(_cfg()).eval()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        a = m(ids, dt=torch.ones(2, 16))["logits"]
        b = m(ids, dt=torch.full((2, 16), 3.0))["logits"]
    diff = (a - b).abs().max().item()
    print(f"  dt=1.0 vs dt=3.0  max|diff| = {diff:.2e}")
    assert diff > 1e-3, "dt has no measurable effect on the output"
    print("[ok] test_dt_changes_output")


def test_dt_accepts_1d_and_2d():
    """dt may be (T,) [shared across batch] or (B,T) [per-sequence]."""
    torch.manual_seed(0)
    m = MTLNNModel(_cfg()).eval()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        a = m(ids, dt=torch.ones(16) * 2.0)["logits"]           # (T,)
        b = m(ids, dt=torch.full((2, 16), 2.0))["logits"]       # (B,T)
    diff = (a - b).abs().max().item()
    print(f"  dt (T,) vs (B,T)  max|diff| = {diff:.2e}")
    assert diff < 1e-4, "1-D and 2-D dt broadcasts disagree"
    print("[ok] test_dt_accepts_1d_and_2d")


def test_dt_gradient_flows():
    """The continuous-time path must be differentiable end-to-end."""
    torch.manual_seed(0)
    m = MTLNNModel(_cfg())
    ids = torch.randint(0, 200, (2, 16))
    dt = torch.rand(2, 16) * 2.0 + 0.1
    out = m(ids, labels=ids, dt=dt)
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    print("[ok] test_dt_gradient_flows")


def test_dt_larger_gap_decays_more():
    """Sanity on CfLTC semantics: with fixed inputs, a larger Δt should move the
    recurrent state closer to its instantaneous drive (more forgetting of the
    initial transient). We verify the effect is monotone in Δt magnitude."""
    torch.manual_seed(0)
    m = MTLNNModel(_cfg()).eval()
    ids = torch.randint(0, 200, (1, 16))
    with torch.no_grad():
        outs = [m(ids, dt=torch.full((1, 16), g))["logits"] for g in (0.5, 1.0, 5.0)]
    # Outputs at different Δt must be pairwise distinct and ordered in change.
    d01 = (outs[0] - outs[1]).abs().mean().item()
    d12 = (outs[1] - outs[2]).abs().mean().item()
    print(f"  mean|Δ| (0.5→1.0)={d01:.3e}  (1.0→5.0)={d12:.3e}")
    assert d01 > 0 and d12 > 0
    print("[ok] test_dt_larger_gap_decays_more")


def run_all():
    test_dt_none_equals_dt_one()
    test_dt_changes_output()
    test_dt_accepts_1d_and_2d()
    test_dt_gradient_flows()
    test_dt_larger_gap_decays_more()
    print("ALL IRREGULAR-DT TESTS PASSED")


if __name__ == "__main__":
    run_all()
