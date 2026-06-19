"""First-principles decode-wall and KV-budget formulas (spec.md §2-§3).

Pure functions, no hardware deps. These are the planner's quantitative core:
_kv_bytes / _ctx_from_budget / _layers_that_fit / _decode_estimate from
spec.md §1.3, made standalone and reusable so the model catalog (HER-9) and
the PLAN rules (HER-12) can both call them.

Unit conventions, matching spec.md exactly:
- KV-cache byte math (kv_cache_bytes / max_context_for_kv_budget) works in
  raw bytes throughout.
- Decode-throughput math (decode_throughput_estimate) mirrors spec.md §2's
  worked numbers, which divide a GB/s bandwidth by a GB-denominated weight
  size (e.g. "170 / 8.0 = 21 tok/s"). Bandwidths are GB/s (decimal, 1e9
  bytes/s) and bytes_per_token is in raw bytes; the conversion is internal.
"""

from .. import contracts

# Bytes per KV element by cache dtype (spec.md §3's "dtype_bytes").
_KV_DTYPE_BYTES: dict[contracts.KvDtype, int] = {
    contracts.KvDtype.FP16: 2,
    contracts.KvDtype.INT8: 1,
    contracts.KvDtype.FP8: 1,
}

_GB = 1_000_000_000  # decimal GB, matching spec.md's GB/s bandwidth figures


def kv_cache_bytes(
    seq_len: int,
    n_layers: int,
    kv_heads: int,
    head_dim: int,
    dtype: contracts.KvDtype,
) -> int:
    """KV-bytes from context/dtype/model geometry (spec.md §3).

    KV_bytes = 2 (K,V) x layers x kv_heads x head_dim x seq_len x dtype_bytes
    """
    dtype_bytes = _KV_DTYPE_BYTES[dtype]
    return 2 * n_layers * kv_heads * head_dim * seq_len * dtype_bytes


def max_context_for_kv_budget(
    budget_bytes: int,
    n_layers: int,
    kv_heads: int,
    head_dim: int,
    dtype: contracts.KvDtype,
) -> int:
    """Context-length from a KV budget — the inverse of kv_cache_bytes."""
    dtype_bytes = _KV_DTYPE_BYTES[dtype]
    bytes_per_token = 2 * n_layers * kv_heads * head_dim * dtype_bytes
    return budget_bytes // bytes_per_token


def layers_that_fit(
    weight_bytes_per_layer: float,
    budget_bytes: int,
    total_layers: int | None = None,
) -> int:
    """How many transformer layers fit resident in a VRAM budget (spec.md §1.3).

    Mirrors _layers_that_fit: VRAM ÷ per-layer weight bytes, capped at the
    model's total layer count when given.
    """
    fit = int(budget_bytes // weight_bytes_per_layer)
    if total_layers is not None:
        fit = min(fit, total_layers)
    return max(fit, 0)


def decode_throughput_estimate(
    bytes_per_token: float,
    mem_bandwidth_gbs: float,
    *,
    gpu_frac: float = 1.0,
    cpu_frac: float = 0.0,
    pcie_bandwidth_gbs: float | None = None,
) -> float:
    """Decode tok/s, blending resident vs. offloaded (PCIe-bound) bandwidth
    (spec.md §2):

        decode tok/s ~ bytes_per_token^-1 / (gpu_frac/bw_vram + cpu_frac/bw_pcie)

    Fully-resident decode (cpu_frac=0) reduces to the simple wall:
    decode throughput ~ memory_bandwidth / active_weight_bytes_per_token.
    """
    bw_vram = mem_bandwidth_gbs * _GB
    time_per_token = bytes_per_token * (gpu_frac / bw_vram)
    if cpu_frac:
        if pcie_bandwidth_gbs is None:
            raise ValueError("pcie_bandwidth_gbs is required when cpu_frac > 0")
        bw_pcie = pcie_bandwidth_gbs * _GB
        time_per_token += bytes_per_token * (cpu_frac / bw_pcie)
    return 1.0 / time_per_token
