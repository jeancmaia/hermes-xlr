"""Synthetic-profile matrix + reference golden test (HER-13).

This is the S1 milestone exit gate. It proves the planner's rules *flip*
correctly across hardware profiles, using synthetic ``HostCapabilities``
records only - **no GPU is required** to run this suite, and every profile
here is a hand-built synthetic (no real product names, no probed values).

Two things live here:

* A **golden test** that pins the reference local profile's (a synthetic 6 GB
  Ampere laptop, Windows, throughput-first) full ``ExecutionPlan``
  field-for-field. Any drift in the planner math, catalog geometry, or
  decision logic turns this red.
* A **synthetic-profile matrix** asserting the three decision flips the
  design calls out: FP8-capable arch -> FP8 KV + draft-target; multi-GPU ->
  layer sharding with no offload; tiny-VRAM + quality-first -> CPU offload
  with the decode cliff surfaced. Each flip is asserted *in contrast* to the
  throughput-first / single-GPU baseline so the rule change is explicit.

All derived numbers below were cross-checked by hand against
``mapper.formulas`` before being pinned - they are deterministic outputs of
pure functions, which is exactly what makes them worth goldening.
"""

from __future__ import annotations

import pytest
from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper import catalog as real_catalog
from hermes_nim_xlr.mapper import planner

# Synthetic sentinel for the driver string. The real probe (HER-8) reports a
# real driver version; these profiles never do, which is what makes them
# recognizably synthetic.
_SYNTH_DRIVER = "0.0.0"


def _gpu(
    name: str,
    arch: contracts.GpuArch,
    cc: tuple[int, int],
    vram_total_mb: int,
    vram_free_mb: int,
    mem_bw: float,
    pcie_bw: float,
) -> contracts.GpuCapabilities:
    """Build a synthetic GpuCapabilities record with capabilities inferred
    from arch/cc.

    Names are synthetic labels (``Reference <arch> <form> <vram>``), not real
    product names - only the arch and the ``Laptop`` substring (which drives
    the planner's shared-display warning) carry behavioral meaning.
    """
    return contracts.GpuCapabilities(
        index=0,
        name=name,
        arch=arch,
        compute_capability=cc,
        vram_total_mb=vram_total_mb,
        vram_free_mb=vram_free_mb,
        mem_bandwidth_gbs=mem_bw,
        pcie_bandwidth_gbs=pcie_bw,
        supports_fp8=arch
        in {
            contracts.GpuArch.ADA,
            contracts.GpuArch.HOPPER,
            contracts.GpuArch.BLACKWELL,
        },
        supports_int8=cc[0] > 7 or (cc[0] == 7 and cc[1] >= 5),
        supports_cuda_graphs=cc[0] >= 7,
        driver_version=_SYNTH_DRIVER,
    )


def _host(
    *gpus: contracts.GpuCapabilities,
    os_name: str = "Windows",
    is_wsl: bool = False,
) -> contracts.HostCapabilities:
    return contracts.HostCapabilities(
        os=os_name,
        is_wsl=is_wsl,
        cpu_ram_gb=32.0,
        container_runtime=None,
        has_nvidia_container_toolkit=False,
        gpus=gpus,
    )


# ---------------------------------------------------------------------------
# Reference profiles shared across the matrix.
# ---------------------------------------------------------------------------

# The reference local profile: a synthetic 6 GB Ampere laptop, Windows. This
# is the worked example from the design doc. The ``Laptop`` substring is
# intentional - it exercises the planner's shared-display-GPU warning path.
_AMPERE_LAPTOP_6GB = _gpu(
    "Reference Ampere Laptop 6GB",
    contracts.GpuArch.AMPERE,
    (8, 6),
    vram_total_mb=6144,
    vram_free_mb=5120,
    mem_bw=170.0,
    pcie_bw=1.969 * 8,
)


# ---------------------------------------------------------------------------
# GOLDEN: the reference profile's full execution plan, pinned field-for-field.
# ---------------------------------------------------------------------------


def test_reference_profile_golden_plan():
    """The reference 6 GB Ampere laptop / throughput-first plan is pinned in
    full.

    A change to *any* planner decision or derived number - KV dtype, weight
    quant, model pick, placement, KV config, decode levers, backend, target
    context, estimated VRAM, estimated decode throughput, rationale, or
    warnings - breaks this test. That is the point: this is the regression
    anchor for the whole PLAN phase.
    """
    plan = planner.plan(_host(_AMPERE_LAPTOP_6GB), catalog_ref=real_catalog)

    # The model the catalog resolves for this budget - fetched rather than
    # reconstructed so the golden pins the planner's *selection* and all
    # derived geometry without duplicating catalog constants.
    nemotron = next(
        m for m in real_catalog.CATALOG if m.repo == "nvidia/Nemotron-4B-Instruct"
    )

    expected = contracts.ExecutionPlan(
        objective=contracts.Objective.THROUGHPUT_FIRST,
        model=nemotron,
        placement=contracts.LayerPlacement(
            total_layers=32,
            gpu_layers=32,
            cpu_offload_layers=0,
            tensor_parallel=1,
            pipeline_parallel=1,
            note="fully GPU-resident",
        ),
        kv=contracts.KvCacheConfig(
            dtype=contracts.KvDtype.INT8,
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.49,
            host_cache_size_bytes=0,
        ),
        levers=contracts.DecodeLevers(
            cuda_graphs=True,
            spec_decode=contracts.SpecDecode.NGRAM,
            draft_model=None,
        ),
        backend=contracts.BackendChoice(
            kind=contracts.BackendKind.LLAMACPP,
            bring_up=contracts.BringUp.NATIVE_WINDOWS,
            serve_endpoint="http://127.0.0.1:8080/v1",
        ),
        target_ctx_tokens=8192,
        est_vram_mb=3480,
        est_decode_tok_s=(29, 47),
        rationale=(
            "INT8 KV: ampere has no native FP8",
            "int4_awq weights for a ~4352 MB budget",
            "nvidia/Nemotron-4B-Instruct (~2200 MB) leaves ~2152 MB for KV",
            "all 32 layers GPU-resident - no offload",
            "n-gram spec-decode: no VRAM for a draft model (zero-cost path)",
            "default native-Windows llama.cpp (CUDA, no WSL2) - "
            "the Windows-tuned local backend",
        ),
        warnings=("shared display GPU - expect eviction; enable prefill warming",),
    )

    assert plan == expected


# ---------------------------------------------------------------------------
# MATRIX: the three decision flips, each asserted in contrast to its baseline.
# ---------------------------------------------------------------------------

# Flip 1: an FP8-capable arch (Ada) flips KV cache to FP8 and enables
# draft-target speculative decoding, where the reference Ampere profile stays
# on INT8 KV + zero-cost n-gram.
_ADA_DESKTOP_24GB = _gpu(
    "Reference Ada Desktop 24GB",
    contracts.GpuArch.ADA,
    (8, 9),
    vram_total_mb=24576,
    vram_free_mb=22000,
    mem_bw=1008.0,
    pcie_bw=1.969 * 16,
)


def test_matrix_fp8_capable_arch_flips_to_fp8_kv_and_draft_target():
    reference = planner.plan(_host(_AMPERE_LAPTOP_6GB), catalog_ref=real_catalog)
    plan = planner.plan(_host(_ADA_DESKTOP_24GB), catalog_ref=real_catalog)

    # The flip: Ada gets FP8 KV and a real draft model.
    assert plan.kv.dtype is contracts.KvDtype.FP8
    assert plan.levers.spec_decode is contracts.SpecDecode.DRAFT_TARGET
    assert plan.levers.draft_model is not None

    # Contrast: the reference Ampere profile stays on INT8 + n-gram.
    assert reference.kv.dtype is contracts.KvDtype.INT8
    assert reference.levers.spec_decode is contracts.SpecDecode.NGRAM
    assert reference.levers.draft_model is None

    # Backend is unchanged - the flip is KV/levers only, not bring-up.
    assert plan.backend is not None
    assert plan.backend.kind is contracts.BackendKind.LLAMACPP
    assert plan.backend.bring_up is contracts.BringUp.NATIVE_WINDOWS


# Flip 2: a multi-GPU host flips to tensor-parallel layer sharding with no
# CPU offload, where the same per-GPU VRAM on a single GPU cannot keep the
# chosen model fully resident without shrinking it.
def test_matrix_multi_gpu_flips_to_layer_sharding_no_offload():
    # Two reference 6 GB Ampere laptops. Combined free VRAM lets the planner
    # reach a larger model than either GPU can hold resident alone.
    host = _host(_AMPERE_LAPTOP_6GB, _AMPERE_LAPTOP_6GB)
    plan = planner.plan(host, catalog_ref=real_catalog)

    # The flip: sharded across both GPUs, nothing on the CPU.
    assert plan.placement.tensor_parallel == 2
    assert plan.placement.cpu_offload_layers == 0
    assert plan.placement.gpu_layers == plan.model.n_layers
    assert "sharded across 2 GPUs" in plan.placement.note

    # Contrast: a single GPU of the same VRAM either stays resident on a
    # smaller model or offloads - it never reports tensor_parallel > 1.
    single = planner.plan(_host(_AMPERE_LAPTOP_6GB), catalog_ref=real_catalog)
    assert single.placement.tensor_parallel == 1


# Flip 3: tiny VRAM under QUALITY_FIRST flips to CPU offload and surfaces the
# decode cliff, where the *same* host under THROUGHPUT_FIRST stays fully
# GPU-resident by picking a smaller model.
_AMPERE_4GB_TINY = _gpu(
    "Reference Ampere 4GB",
    contracts.GpuArch.AMPERE,
    (8, 6),
    vram_total_mb=4096,
    vram_free_mb=3500,
    mem_bw=170.0,
    pcie_bw=1.969 * 8,
)


def test_matrix_tiny_vram_quality_first_flips_to_cpu_offload_with_penalty():
    throughput_first = planner.plan(_host(_AMPERE_4GB_TINY), catalog_ref=real_catalog)
    quality_first = planner.plan(
        _host(_AMPERE_4GB_TINY),
        catalog_ref=real_catalog,
        objective=contracts.Objective.QUALITY_FIRST,
    )

    # Contrast: throughput-first stays resident (no offload).
    assert throughput_first.placement.cpu_offload_layers == 0

    # The flip: quality-first accepts CPU offload to run a stronger model.
    assert quality_first.objective is contracts.Objective.QUALITY_FIRST
    assert quality_first.placement.cpu_offload_layers > 0
    assert quality_first.placement.gpu_layers < quality_first.placement.total_layers
    assert quality_first.placement.tensor_parallel == 1

    # The throughput penalty is surfaced - the decode cliff multiplier must
    # appear in the warnings, not be silently swallowed.
    cliff_warnings = [w for w in quality_first.warnings if "layers on CPU" in w]
    assert cliff_warnings, "expected a CPU-offload decode-cliff warning"
    assert "slower" in cliff_warnings[0]


# ---------------------------------------------------------------------------
# Guardrail: the matrix never touches real hardware - every GPU record here
# is synthetic. This assert documents that invariant and protects against an
# accidental real-probe import sneaking into the fixtures.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    [
        _AMPERE_LAPTOP_6GB,
        _ADA_DESKTOP_24GB,
        _AMPERE_4GB_TINY,
    ],
    ids=["ampere-laptop-6gb", "ada-desktop-24gb", "ampere-4gb-tiny"],
)
def test_matrix_profiles_are_synthetic(profile: contracts.GpuCapabilities):
    """Every GPU exercised here is hand-built, not probed from the host.

    The driver string is the synthetic sentinel: the real probe (HER-8) would
    never report this exact constant. If that changes, this test fails loud.
    """
    assert profile.driver_version == _SYNTH_DRIVER
    assert isinstance(profile, contracts.GpuCapabilities)
