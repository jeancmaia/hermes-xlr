# Model: Nemotron-Mini-4B-Instruct (Q4_K_M GGUF)

**Chosen for:** S0.5 bring-up spike (HER-3)

## Selection rationale

| Criteria | Value |
|---|---|
| **HF repo** | `nvidia/Nemotron-Mini-4B-Instruct` |
| **GGUF source** | `bartowski/Nemotron-Mini-4B-Instruct-GGUF` |
| **GGUF variant** | `Nemotron-Mini-4B-Instruct-Q4_K_M.gguf` |
| **Parameters** | 4.0B |
| **Weight size** | ~2.2 GB at Q4_K_M |
| **Context length** | 4096 tokens |
| **Architecture** | Transformer decoder (Nemotron-4), GQA, RoPE |
| **License** | [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf) — commercial use allowed |
| **Why this model** | Reference model from the design spec (`spec.md` §1.4). Fits the ~2750 MB weight budget on a 6 GB GPU (~5 GB usable) with ~2 GB headroom for KV cache and activations. 32 layers, all GPU-resident with no offload. |

## Expected VRAM budget (RTX 3050 6 GB Laptop)

| Component | Size |
|---|---|
| Weights (Q4_K_M GGUF) | ~2200 MB |
| KV cache (INT8, 4096 ctx) | ~256 MB |
| Safety/overhead headroom | ~768 MB |
| **Total estimated** | ~3224 MB |
| Usable VRAM | ~4352 MB (5120 MB free - 768 MB safety) |
| **Headroom** | ~1128 MB |

## Download

```powershell
# Requires: curl or Invoke-WebRequest
.\download.ps1
```

## Verification

After download, confirm the model loads with `llama-cli`:

```powershell
llama-cli --model models\nemotron-mini-4b-instruct-q4_k_m.gguf -p "Hello" -n 10
```