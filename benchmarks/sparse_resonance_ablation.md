# Sparse Resonance Ablation

Compares dense tau-scale computation against optional sparse top-k resonance.

| Mode | Top-k | Selected scale ratio | Time (s) | Tok/s | Mean div vs dense | Max div vs dense |
|---|---:|---:|---:|---:|---:|---:|
| `dense` | 5 | 1.000 | 0.0273 | 3659.9 | 0 | 0 |
| `sparse` | 1 | 0.200 | 0.0173 | 5771.3 | 0.00261 | 0.107 |
| `sparse` | 2 | 0.400 | 0.0156 | 6402.9 | 0.00331 | 0.0796 |
| `sparse` | 3 | 0.600 | 0.0206 | 4865.4 | 0.00171 | 0.0699 |

## Caveats

- Sparse top-k selection is intentionally approximate and can change outputs.
- The current implementation uses PyTorch tensor indexing and top-k over tau scales, not a custom CUDA/Triton kernel.
- Speedups depend on sequence length, device, and whether top-k selection overhead dominates the skipped scale work.
