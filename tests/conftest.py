"""Shared pytest fixtures; profile-specific ones live under tests/fixtures/."""

from __future__ import annotations

import pytest
from hermes_nim_xlr.contracts import (
    BackendChoice,
    BackendKind,
    BringUp,
    DecodeLevers,
    ExecutionPlan,
    KvCacheConfig,
    KvDtype,
    LayerPlacement,
    ModelChoice,
    Objective,
    SpecDecode,
    WeightQuant,
)


@pytest.fixture
def package_name() -> str:
    """Placeholder proving the fixtures wiring; replace as real fixtures land."""
    return "hermes_nim_xlr"


@pytest.fixture
def sample_plan() -> ExecutionPlan:
    return ExecutionPlan(
        objective=Objective.THROUGHPUT_FIRST,
        model=ModelChoice(
            repo="QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF",
            params_b=3.2,
            weight_quant=WeightQuant.INT4_AWQ,
            est_weight_mb=1800,
            n_layers=28,
            kv_heads=8,
            head_dim=128,
            max_context_tokens=8192,
        ),
        placement=LayerPlacement(
            total_layers=28,
            gpu_layers=28,
            cpu_offload_layers=0,
            tensor_parallel=1,
            pipeline_parallel=1,
            note="fully GPU-resident",
        ),
        kv=KvCacheConfig(
            dtype=KvDtype.INT8,
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.35,
            host_cache_size_bytes=0,
        ),
        levers=DecodeLevers(
            cuda_graphs=False,
            spec_decode=SpecDecode.NONE,
            draft_model=None,
        ),
        backend=BackendChoice(
            kind=BackendKind.LLAMACPP,
            bring_up=BringUp.NATIVE_WINDOWS,
            serve_endpoint="http://127.0.0.1:8080/v1",
        ),
        target_ctx_tokens=4096,
        est_vram_mb=2800,
        est_decode_tok_s=(30, 50),
        rationale=("test fixture",),
        warnings=(),
    )
