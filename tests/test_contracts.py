import dataclasses

import pytest
from hermes_nim_xlr.contracts import (
    BackendChoice,
    BackendKind,
    BringUp,
    DecodeLevers,
    ExecutionPlan,
    GpuArch,
    GpuCapabilities,
    HostCapabilities,
    KvCacheConfig,
    KvDtype,
    LayerPlacement,
    ModelChoice,
    Objective,
    SpecDecode,
    WeightQuant,
)


def _make_gpu() -> GpuCapabilities:
    return GpuCapabilities(
        index=0,
        name="NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        arch=GpuArch.AMPERE,
        compute_capability=(8, 6),
        vram_total_mb=6144,
        vram_free_mb=5000,
        mem_bandwidth_gbs=170.0,
        pcie_bandwidth_gbs=16.0,
        supports_fp8=False,
        supports_int8=True,
        supports_cuda_graphs=True,
        driver_version="551.23",
    )


def _make_host() -> HostCapabilities:
    return HostCapabilities(
        os="Windows",
        is_wsl=False,
        cpu_ram_gb=32.0,
        container_runtime=None,
        has_nvidia_container_toolkit=False,
        gpus=(_make_gpu(),),
    )


def _make_plan() -> ExecutionPlan:
    model = ModelChoice(
        repo="nvidia/Nemotron-4B-Instruct",
        params_b=4.0,
        weight_quant=WeightQuant.INT4_AWQ,
        est_weight_mb=2200,
        n_layers=32,
        kv_heads=8,
        head_dim=128,
        max_context_tokens=8192,
    )
    placement = LayerPlacement(
        total_layers=32,
        gpu_layers=32,
        cpu_offload_layers=0,
        tensor_parallel=1,
        pipeline_parallel=1,
        note="fully GPU-resident",
    )
    kv = KvCacheConfig(
        dtype=KvDtype.INT8,
        enable_block_reuse=True,
        free_gpu_memory_fraction=0.45,
        host_cache_size_bytes=0,
    )
    levers = DecodeLevers(
        cuda_graphs=True, spec_decode=SpecDecode.NGRAM, draft_model=None
    )
    backend = BackendChoice(
        kind=BackendKind.LLAMACPP,
        bring_up=BringUp.NATIVE_WINDOWS,
        serve_endpoint="http://127.0.0.1:8080/v1",
    )
    return ExecutionPlan(
        objective=Objective.THROUGHPUT_FIRST,
        model=model,
        placement=placement,
        kv=kv,
        levers=levers,
        backend=backend,
        target_ctx_tokens=8192,
        est_vram_mb=4100,
        est_decode_tok_s=(30, 50),
        rationale=("INT8 KV: ampere has no native FP8",),
        warnings=("shared display GPU — expect eviction",),
    )


def test_host_capabilities_construction():
    host = _make_host()
    assert host.gpus[0].arch is GpuArch.AMPERE


def test_execution_plan_construction():
    plan = _make_plan()
    assert plan.model.repo == "nvidia/Nemotron-4B-Instruct"
    assert plan.kv.dtype is KvDtype.INT8


def test_execution_plan_round_trips_through_asdict():
    plan = _make_plan()
    as_dict = dataclasses.asdict(plan)
    assert as_dict["objective"] == Objective.THROUGHPUT_FIRST
    assert as_dict["model"]["repo"] == "nvidia/Nemotron-4B-Instruct"
    rebuilt = ExecutionPlan(
        objective=as_dict["objective"],
        model=ModelChoice(**as_dict["model"]),
        placement=LayerPlacement(**as_dict["placement"]),
        kv=KvCacheConfig(**as_dict["kv"]),
        levers=DecodeLevers(**as_dict["levers"]),
        backend=BackendChoice(**as_dict["backend"]),
        target_ctx_tokens=as_dict["target_ctx_tokens"],
        est_vram_mb=as_dict["est_vram_mb"],
        est_decode_tok_s=as_dict["est_decode_tok_s"],
        rationale=as_dict["rationale"],
        warnings=as_dict["warnings"],
    )
    assert rebuilt == plan


@pytest.mark.parametrize(
    "field, value",
    [("est_vram_mb", 9999), ("target_ctx_tokens", 1)],
)
def test_execution_plan_is_frozen(field, value):
    plan = _make_plan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(plan, field, value)
