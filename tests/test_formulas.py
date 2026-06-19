"""Hand-worked and property tests for the decode-wall / KV-budget formulas
(spec.md §2-§3), grounding HER-10 against the spec's own worked numbers.
"""

import pytest

from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper import formulas

# ---------------------------------------------------------------------------
# kv_cache_bytes — spec.md §3 worked example:
#   4B model, 32 layers, 8 KV heads (GQA), head_dim 128, 8K context, FP16:
#   2 x 32 x 8 x 128 x 8192 x 2 = 1.07 GB
#   same with FP8 KV: = 0.54 GB
# ---------------------------------------------------------------------------


def test_kv_cache_bytes_matches_spec_fp16_worked_example():
    # 2 * 32 * 8 * 128 * 8192 * 2 = 1_073_741_824 bytes = exactly 1 GiB,
    # which is spec.md's "~1.07 GB" (spec uses decimal-GB shorthand).
    result = formulas.kv_cache_bytes(
        seq_len=8192,
        n_layers=32,
        kv_heads=8,
        head_dim=128,
        dtype=contracts.KvDtype.FP16,
    )
    assert result == 1_073_741_824
    assert result == pytest.approx(1.07e9, rel=0.01)


def test_kv_cache_bytes_matches_spec_fp8_worked_example():
    # FP8 (1 byte/element) is exactly half of FP16 (2 bytes/element) at the
    # same geometry: spec.md says "≈ 0.54 GB".
    fp16_bytes = formulas.kv_cache_bytes(
        seq_len=8192,
        n_layers=32,
        kv_heads=8,
        head_dim=128,
        dtype=contracts.KvDtype.FP16,
    )
    fp8_bytes = formulas.kv_cache_bytes(
        seq_len=8192,
        n_layers=32,
        kv_heads=8,
        head_dim=128,
        dtype=contracts.KvDtype.FP8,
    )
    assert fp8_bytes == fp16_bytes // 2
    assert fp8_bytes == pytest.approx(0.54e9, rel=0.01)


def test_kv_cache_bytes_int8_same_width_as_fp8():
    # INT8 and FP8 are both 1 byte/element per spec.md §3/§4.2.
    int8_bytes = formulas.kv_cache_bytes(
        seq_len=8192,
        n_layers=32,
        kv_heads=8,
        head_dim=128,
        dtype=contracts.KvDtype.INT8,
    )
    fp8_bytes = formulas.kv_cache_bytes(
        seq_len=8192,
        n_layers=32,
        kv_heads=8,
        head_dim=128,
        dtype=contracts.KvDtype.FP8,
    )
    assert int8_bytes == fp8_bytes


# ---------------------------------------------------------------------------
# max_context_for_kv_budget — the inverse of kv_cache_bytes.
# ---------------------------------------------------------------------------


def test_max_context_for_kv_budget_inverts_kv_cache_bytes():
    geometry = dict(n_layers=32, kv_heads=8, head_dim=128, dtype=contracts.KvDtype.FP16)
    budget = formulas.kv_cache_bytes(seq_len=8192, **geometry)
    ctx = formulas.max_context_for_kv_budget(budget_bytes=budget, **geometry)
    assert ctx == 8192


def test_max_context_monotonically_decreases_as_kv_dtype_widens():
    # Same byte budget, wider dtype (more bytes/element) => fewer tokens fit.
    geometry = dict(n_layers=32, kv_heads=8, head_dim=128)
    budget = 2_000_000_000  # 2 GB, arbitrary fixed budget
    ctx_fp16 = formulas.max_context_for_kv_budget(
        budget_bytes=budget, dtype=contracts.KvDtype.FP16, **geometry
    )
    ctx_int8 = formulas.max_context_for_kv_budget(
        budget_bytes=budget, dtype=contracts.KvDtype.INT8, **geometry
    )
    assert ctx_fp16 < ctx_int8


# ---------------------------------------------------------------------------
# layers_that_fit — spec.md §1.3's _layers_that_fit.
# ---------------------------------------------------------------------------


def test_layers_that_fit_hand_worked():
    # 100 MB/layer, 350 MB budget => 3 whole layers fit (300 <= 350 < 400).
    fit = formulas.layers_that_fit(weight_bytes_per_layer=100, budget_bytes=350)
    assert fit == 3


def test_layers_that_fit_caps_at_total_layers():
    # Budget would allow 10 layers, but the model only has 4.
    fit = formulas.layers_that_fit(
        weight_bytes_per_layer=100, budget_bytes=1000, total_layers=4
    )
    assert fit == 4


def test_layers_that_fit_zero_budget_fits_nothing():
    fit = formulas.layers_that_fit(weight_bytes_per_layer=100, budget_bytes=0)
    assert fit == 0


# ---------------------------------------------------------------------------
# decode_throughput_estimate — spec.md §2's worked numbers:
#   4B model @ FP16  (~8 GB):  170 / 8.0 = 21 tok/s ceiling
#   4B model @ INT4  (~2.2 GB): 170 / 2.2 = 77 tok/s ceiling
# ---------------------------------------------------------------------------


def test_decode_throughput_matches_spec_fp16_worked_example():
    tok_s = formulas.decode_throughput_estimate(
        bytes_per_token=8.0e9, mem_bandwidth_gbs=170.0
    )
    assert tok_s == pytest.approx(21.25, rel=1e-9)


def test_decode_throughput_matches_spec_int4_worked_example():
    tok_s = formulas.decode_throughput_estimate(
        bytes_per_token=2.2e9, mem_bandwidth_gbs=170.0
    )
    assert tok_s == pytest.approx(170.0 / 2.2, rel=1e-9)


def test_decode_throughput_blended_hand_worked():
    # Hand worked: bytes_per_token=2e9, bw_vram=200 GB/s, bw_pcie=20 GB/s,
    # half the layers offloaded (gpu_frac=cpu_frac=0.5).
    #   time/token = 2e9 * (0.5/200e9 + 0.5/20e9)
    #              = 2e9 * (2.5e-12 + 2.5e-11) = 2e9 * 2.75e-11 = 0.055 s
    #   tok/s = 1 / 0.055 = 18.181818...
    tok_s = formulas.decode_throughput_estimate(
        bytes_per_token=2.0e9,
        mem_bandwidth_gbs=200.0,
        gpu_frac=0.5,
        cpu_frac=0.5,
        pcie_bandwidth_gbs=20.0,
    )
    assert tok_s == pytest.approx(1 / 0.055, rel=1e-9)


def test_decode_throughput_offload_lowers_estimate_vs_fully_resident():
    # spec.md §2 offload corollary: PCIe is far slower than VRAM bandwidth,
    # so any CPU-offloaded fraction must lower the estimate vs. fully
    # resident decode at the same bytes_per_token.
    bytes_per_token = 2.2e9
    resident = formulas.decode_throughput_estimate(
        bytes_per_token=bytes_per_token, mem_bandwidth_gbs=170.0
    )
    offloaded = formulas.decode_throughput_estimate(
        bytes_per_token=bytes_per_token,
        mem_bandwidth_gbs=170.0,
        gpu_frac=0.5,
        cpu_frac=0.5,
        pcie_bandwidth_gbs=24.0,
    )
    assert offloaded < resident


def test_decode_throughput_requires_pcie_bandwidth_when_offloading():
    with pytest.raises(ValueError):
        formulas.decode_throughput_estimate(
            bytes_per_token=1.0e9, mem_bandwidth_gbs=170.0, cpu_frac=0.5
        )
