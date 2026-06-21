"""Tests for the deterministic planner (HER-12 / spec.md §1.3).

Builds synthetic HostCapabilities records so the planner can be exercised
without real hardware, then checks the structural decisions (KV dtype,
weight quant, placement, backend, rationale, warnings) against the spec's
expectations.
"""

import pytest

from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper import catalog as real_catalog
from hermes_nim_xlr.mapper import planner


def _gpu(
    name: str,
    arch: contracts.GpuArch,
    cc: tuple[int, int],
    vram_total_mb: int,
    vram_free_mb: int,
    mem_bw: float,
    pcie_bw: float,
) -> contracts.GpuCapabilities:
    """Build a GpuCapabilities record with all capabilities inferred from arch."""
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
        driver_version="555.99",
    )


def _host(
    *gpus: contracts.GpuCapabilities,
    os_name: str = "Windows",
    is_wsl: bool = False,
    container_runtime: str | None = None,
    has_toolkit: bool = False,
) -> contracts.HostCapabilities:
    return contracts.HostCapabilities(
        os=os_name,
        is_wsl=is_wsl,
        cpu_ram_gb=32.0,
        container_runtime=container_runtime,
        has_nvidia_container_toolkit=has_toolkit,
        gpus=gpus,
    )


# ---------------------------------------------------------------------------
# Reference golden: RTX 3050 6 GB Laptop, Windows, throughput-first.
# This is the worked example from spec.md §1.4, checked structurally.
# ---------------------------------------------------------------------------


def test_reference_rtx_3050_emits_expected_plan():
    rtx3050 = _gpu(
        "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        contracts.GpuArch.AMPERE,
        (8, 6),
        vram_total_mb=6144,
        vram_free_mb=5120,
        mem_bw=170.0,
        pcie_bw=1.969 * 8,
    )
    host = _host(rtx3050)

    plan = planner.plan(host, catalog_ref=real_catalog)

    assert plan.objective is contracts.Objective.THROUGHPUT_FIRST
    assert plan.model.repo == "nvidia/Nemotron-4B-Instruct"
    assert plan.model.weight_quant is contracts.WeightQuant.INT4_AWQ
    assert plan.placement.cpu_offload_layers == 0
    assert plan.placement.gpu_layers == 32
    assert plan.placement.tensor_parallel == 1

    assert plan.kv.dtype is contracts.KvDtype.INT8
    assert plan.kv.enable_block_reuse is True
    assert plan.kv.host_cache_size_bytes == 0

    assert plan.levers.cuda_graphs is True
    assert plan.levers.spec_decode is contracts.SpecDecode.NGRAM
    assert plan.levers.draft_model is None

    assert plan.backend.kind is contracts.BackendKind.LLAMACPP
    assert plan.backend.bring_up is contracts.BringUp.NATIVE_WINDOWS

    assert plan.target_ctx_tokens <= plan.model.max_context_tokens
    assert plan.est_vram_mb > 0
    assert 0 < plan.est_decode_tok_s[0] <= plan.est_decode_tok_s[1]

    rationale_text = "\n".join(plan.rationale)
    assert "INT8 KV" in rationale_text
    assert "int4_awq weights" in rationale_text
    assert "Nemotron-4B-Instruct" in rationale_text
    assert "layers GPU-resident" in rationale_text
    assert "n-gram spec-decode" in rationale_text
    assert "native-Windows llama.cpp" in rationale_text

    assert any("shared display GPU" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Ada / Hopper + native Linux → FP8 KV, draft-target, TRT-LLM native.
# ---------------------------------------------------------------------------


def test_ada_desktop_linux_prefers_fp8_and_draft_target():
    rtx4090 = _gpu(
        "NVIDIA GeForce RTX 4090",
        contracts.GpuArch.ADA,
        (8, 9),
        vram_total_mb=24576,
        vram_free_mb=22000,
        mem_bw=1008.0,
        pcie_bw=1.969 * 16,
    )
    host = _host(
        rtx4090,
        os_name="Linux",
        is_wsl=False,
        container_runtime="docker",
        has_toolkit=True,
    )

    plan = planner.plan(host, catalog_ref=real_catalog)

    assert plan.kv.dtype is contracts.KvDtype.FP8
    assert plan.model.weight_quant is contracts.WeightQuant.INT8
    assert plan.levers.spec_decode is contracts.SpecDecode.DRAFT_TARGET
    assert plan.levers.draft_model is not None
    assert plan.backend.kind is contracts.BackendKind.TRTLLM
    assert plan.backend.bring_up is contracts.BringUp.NATIVE_LINUX


# ---------------------------------------------------------------------------
# Multi-GPU → layer sharding, no CPU offload.
#
# Per-GPU VRAM is intentionally tight enough that the selected model cannot
# fit fully on a single GPU, forcing the planner to shard instead of picking
# a fully-resident smaller model.
# ---------------------------------------------------------------------------


def test_multi_gpu_profile_chooses_sharding_not_offload():
    small_ampere = _gpu(
        "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        contracts.GpuArch.AMPERE,
        (8, 6),
        vram_total_mb=6144,
        vram_free_mb=5120,
        mem_bw=170.0,
        pcie_bw=1.969 * 8,
    )
    host = _host(small_ampere, small_ampere, os_name="Linux", is_wsl=False)

    plan = planner.plan(host, catalog_ref=real_catalog)

    assert plan.placement.cpu_offload_layers == 0
    assert plan.placement.tensor_parallel == 2
    assert plan.placement.gpu_layers == plan.model.n_layers
    assert "sharded across 2 GPUs" in plan.placement.note


# ---------------------------------------------------------------------------
# Pure/deterministic: identical input → identical plan.
# ---------------------------------------------------------------------------


def test_plan_is_deterministic():
    rtx3050 = _gpu(
        "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        contracts.GpuArch.AMPERE,
        (8, 6),
        vram_total_mb=6144,
        vram_free_mb=5120,
        mem_bw=170.0,
        pcie_bw=1.969 * 8,
    )
    host = _host(rtx3050)

    first = planner.plan(host, catalog_ref=real_catalog)
    second = planner.plan(host, catalog_ref=real_catalog)

    assert first == second


# ---------------------------------------------------------------------------
# QUALITY_FIRST CPU-offload branch.
# QUALITY_FIRST ignores the 0.55 weight budget so it can pick the strongest
# model for the quant; layers that do not fit on the primary GPU are offloaded
# to host RAM, and a warning is emitted.
# ---------------------------------------------------------------------------


def test_quality_first_can_accept_cpu_offload():
    rtx3050 = _gpu(
        "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        contracts.GpuArch.AMPERE,
        (8, 6),
        vram_total_mb=6144,
        vram_free_mb=5120,
        mem_bw=170.0,
        pcie_bw=1.969 * 8,
    )
    host = _host(rtx3050)

    plan = planner.plan(
        host,
        catalog_ref=real_catalog,
        objective=contracts.Objective.QUALITY_FIRST,
    )

    assert plan.objective is contracts.Objective.QUALITY_FIRST
    assert plan.placement.cpu_offload_layers > 0
    assert any("layers on CPU" in w for w in plan.warnings)


# ---------------------------------------------------------------------------
# Windows + prefer_performance + toolkit → WSL2 TRT-LLM path.
# ---------------------------------------------------------------------------


def test_windows_prefer_performance_selects_wsl2_trtllm():
    rtx4090 = _gpu(
        "NVIDIA GeForce RTX 4090",
        contracts.GpuArch.ADA,
        (8, 9),
        vram_total_mb=24576,
        vram_free_mb=22000,
        mem_bw=1008.0,
        pcie_bw=1.969 * 16,
    )
    host = _host(
        rtx4090,
        os_name="Windows",
        container_runtime="docker",
        has_toolkit=True,
    )

    plan = planner.plan(
        host,
        catalog_ref=real_catalog,
        prefer_performance=True,
    )

    assert plan.backend.kind is contracts.BackendKind.TRTLLM
    assert plan.backend.bring_up is contracts.BringUp.WSL2_DOCKER


# ---------------------------------------------------------------------------
# Host with no GPU should raise clearly.
# ---------------------------------------------------------------------------


def test_plan_with_no_gpu_raises():
    host = contracts.HostCapabilities(
        os="Windows",
        is_wsl=False,
        cpu_ram_gb=32.0,
        container_runtime=None,
        has_nvidia_container_toolkit=False,
        gpus=(),
    )
    with pytest.raises(ValueError):
        planner.plan(host, catalog_ref=real_catalog)


# ---------------------------------------------------------------------------
# Each branch must contribute a rationale entry.
# ---------------------------------------------------------------------------


def test_rationale_is_auditable():
    rtx3050 = _gpu(
        "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        contracts.GpuArch.AMPERE,
        (8, 6),
        vram_total_mb=6144,
        vram_free_mb=5120,
        mem_bw=170.0,
        pcie_bw=1.969 * 8,
    )
    host = _host(rtx3050)

    plan = planner.plan(host, catalog_ref=real_catalog)

    assert len(plan.rationale) >= 5
    for line in plan.rationale:
        assert isinstance(line, str) and line.strip()
