"""Deterministic execution-plan generator - the PLAN phase.

``plan()`` is a pure function from ``HostCapabilities`` to
``ExecutionPlan``. It applies five auditable rules and records the
reasoning behind every branch in ``rationale``. There is no per-turn
entropy, no probe side-effects, and no dependence on anything but the
supplied ``HostCapabilities`` record and catalog.
"""

from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper import catalog, formulas

_SAFETY_MB = 768  # activations + framework/runtime overhead headroom
_MB = 1024 * 1024

# Decode-throughput range is inherently noisy (cache hit rate, launch
# overhead, actual power state). The reference 3050 anchors estimates like
# "~30-50 tok/s". We express the range as a conservative band around the
# first-principles wall so the numbers are honest without pretending to be
# exact.
_DECODE_EFFICIENCY_LOW = 0.40
_DECODE_EFFICIENCY_HIGH = 0.65


def _primary_gpu(host: contracts.HostCapabilities) -> contracts.GpuCapabilities:
    """Return the GPU with the most free VRAM.

    CPU-only hosts error until a CPU backend ships.
    """
    if not host.gpus:
        raise ValueError("no GPU detected; CPU-only execution is not supported yet")
    return max(host.gpus, key=lambda g: (g.vram_free_mb, -g.index))


def _kv_bytes(
    target_ctx: int, kv_dtype: contracts.KvDtype, model: contracts.ModelChoice
) -> int:
    """KV-cache bytes for ``target_ctx`` tokens."""
    return formulas.kv_cache_bytes(
        seq_len=target_ctx,
        n_layers=model.n_layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        dtype=kv_dtype,
    )


def _ctx_from_budget(
    kv_budget_mb: int, kv_dtype: contracts.KvDtype, model: contracts.ModelChoice
) -> int:
    """Maximum context length invertable from the KV budget.

    The result is clamped to the model's own context ceiling.
    """
    safe_budget_mb = max(kv_budget_mb, 0)
    ctx = formulas.max_context_for_kv_budget(
        budget_bytes=safe_budget_mb * _MB,
        n_layers=model.n_layers,
        kv_heads=model.kv_heads,
        head_dim=model.head_dim,
        dtype=kv_dtype,
    )
    return min(ctx, model.max_context_tokens)


def _layers_that_fit(model: contracts.ModelChoice, vram_budget_mb: int) -> int:
    """How many transformer layers fit inside ``vram_budget_mb``.

    This is the layer-granularity placement check. Using a plain VRAM
    budget (rather than the post-weight KV budget) keeps the decision
    physically meaningful: if the model's total weight exceeds the VRAM we
    are willing to allocate, layers spill.
    """
    weight_bytes_per_layer = (model.est_weight_mb * _MB) / model.n_layers
    return formulas.layers_that_fit(
        weight_bytes_per_layer=weight_bytes_per_layer,
        budget_bytes=vram_budget_mb * _MB,
        total_layers=model.n_layers,
    )


def _resident(model: contracts.ModelChoice) -> contracts.LayerPlacement:
    """Fully GPU-resident placement."""
    return contracts.LayerPlacement(
        total_layers=model.n_layers,
        gpu_layers=model.n_layers,
        cpu_offload_layers=0,
        tensor_parallel=1,
        pipeline_parallel=1,
        note="fully GPU-resident",
    )


def _shard_across_gpus(
    model: contracts.ModelChoice, gpus: tuple[contracts.GpuCapabilities, ...]
) -> contracts.LayerPlacement:
    """Tensor-parallel placement across all detected GPUs."""
    return contracts.LayerPlacement(
        total_layers=model.n_layers,
        gpu_layers=model.n_layers,
        cpu_offload_layers=0,
        tensor_parallel=len(gpus),
        pipeline_parallel=1,
        note=f"sharded across {len(gpus)} GPUs (no CPU offload)",
    )


def _cliff(gpu: contracts.GpuCapabilities) -> float:
    """PCIe-vs-VRAM slowdown multiplier for CPU-offloaded layers.

    Returns a conservative fallback when either bandwidth is unknown.
    """
    if gpu.mem_bandwidth_gbs and gpu.pcie_bandwidth_gbs:
        return round(gpu.mem_bandwidth_gbs / gpu.pcie_bandwidth_gbs, 1)
    return 8.0


def _decode_estimate(
    gpu: contracts.GpuCapabilities,
    model: contracts.ModelChoice,
    placement: contracts.LayerPlacement,
) -> tuple[int, int]:
    """Decode throughput band, blending VRAM and PCIe bandwidth if layers are
    offloaded.
    """
    if gpu.mem_bandwidth_gbs is None:
        return (0, 0)

    bytes_per_token = model.est_weight_mb * _MB
    cpu_frac = placement.cpu_offload_layers / placement.total_layers
    raw = formulas.decode_throughput_estimate(
        bytes_per_token=bytes_per_token,
        mem_bandwidth_gbs=gpu.mem_bandwidth_gbs,
        cpu_frac=cpu_frac,
        pcie_bandwidth_gbs=gpu.pcie_bandwidth_gbs if cpu_frac else None,
    )
    return (
        int(raw * _DECODE_EFFICIENCY_LOW),
        int(raw * _DECODE_EFFICIENCY_HIGH),
    )


def _select_model_for_objective(
    catalog_ref: object,
    weight_quant: contracts.WeightQuant,
    weight_budget: int,
    objective: contracts.Objective,
) -> contracts.ModelChoice:
    """Pick the model according to the speed-vs-quality policy.

    ``THROUGHPUT_FIRST`` respects the 0.55 VRAM budget: it wants the largest
    model that still leaves headroom for the KV cache. ``QUALITY_FIRST``
    ignores that cap and takes the strongest model available for the chosen
    quant, accepting that layers may have to CPU-offload.
    """
    if objective is contracts.Objective.QUALITY_FIRST:
        candidates = [m for m in catalog_ref.CATALOG if m.weight_quant is weight_quant]
        if not candidates:
            raise ValueError(f"no {weight_quant.value} model in catalog")
        return max(candidates, key=lambda m: m.est_weight_mb)
    return catalog_ref.largest_fitting(weight_quant, budget_mb=weight_budget)


def plan(
    host: contracts.HostCapabilities,
    catalog_ref: object = catalog,
    objective: contracts.Objective = contracts.Objective.THROUGHPUT_FIRST,
) -> contracts.ExecutionPlan:
    """Generate a deterministic execution plan from host capabilities.

    Every branch appends a human-readable rationale line;
    laptop/shared-display GPUs generate a warning. The function is pure:
    identical input always yields identical output.

    Args:
        host: The detected host/GPU capabilities.
        catalog_ref: A catalog module/object exposing ``largest_fitting``,
            ``largest_fully_fitting``, ``draft_for`` and ``CATALOG``.
        objective: ``THROUGHPUT_FIRST`` avoids CPU offload;
            ``QUALITY_FIRST`` accepts a PCIe cliff to run a stronger model.

    Returns:
        An immutable ``ExecutionPlan`` ready for the backend seam.
    """
    why: list[str] = []
    warnings: list[str] = []

    primary_gpu = _primary_gpu(host)

    if len(host.gpus) > 1:
        total_vram_free = sum(g.vram_free_mb for g in host.gpus)
        usable_total = max(0, total_vram_free - _SAFETY_MB)
    else:
        usable_total = max(0, primary_gpu.vram_free_mb - _SAFETY_MB)

    primary_usable = max(0, primary_gpu.vram_free_mb - _SAFETY_MB)
    weight_budget = int(usable_total * 0.55)

    if primary_gpu.supports_fp8:
        kv_dtype = contracts.KvDtype.FP8
        why.append("FP8 KV: native FP8 on this arch")
    elif primary_gpu.supports_int8:
        kv_dtype = contracts.KvDtype.INT8
        why.append(f"INT8 KV: {primary_gpu.arch.value} has no native FP8")
    else:
        kv_dtype = contracts.KvDtype.FP16
        warnings.append("no INT8/FP8 support detected - KV cache left uncompressed")

    weight_quant = (
        contracts.WeightQuant.INT4_AWQ
        if usable_total < 12_000
        else contracts.WeightQuant.INT8
    )
    why.append(f"{weight_quant.value} weights for a ~{usable_total} MB budget")

    model = _select_model_for_objective(
        catalog_ref, weight_quant, weight_budget, objective
    )
    kv_budget = usable_total - model.est_weight_mb
    why.append(
        f"{model.repo} (~{model.est_weight_mb} MB) leaves ~{kv_budget} MB for KV"
    )

    fit = _layers_that_fit(model, primary_usable)
    if fit >= model.n_layers:
        placement = _resident(model)
        why.append(f"all {model.n_layers} layers GPU-resident - no offload")
    elif len(host.gpus) > 1:
        placement = _shard_across_gpus(model, host.gpus)
        why.append(f"sharded across {len(host.gpus)} GPUs (no CPU offload)")
    elif objective is contracts.Objective.THROUGHPUT_FIRST:
        model = catalog_ref.largest_fully_fitting(
            weight_quant, budget_mb=primary_usable
        )
        placement = _resident(model)
        why.append(
            f"throughput-first: fully-resident {model.repo} chosen over CPU offload"
        )
    else:
        placement = contracts.LayerPlacement(
            total_layers=model.n_layers,
            gpu_layers=fit,
            cpu_offload_layers=model.n_layers - fit,
            tensor_parallel=1,
            pipeline_parallel=1,
            note="PCIe-bound CPU offload - decode cliff",
        )
        warnings.append(
            f"{placement.cpu_offload_layers}/{model.n_layers} layers on CPU - "
            f"decode ~{_cliff(primary_gpu)}x slower"
        )

    draft_repo = catalog_ref.draft_for(model)
    draft_mb = next(
        m.est_weight_mb
        for m in catalog_ref.CATALOG
        if m.repo == draft_repo and m.weight_quant is model.weight_quant
    )
    if model.est_weight_mb + draft_mb <= weight_budget:
        spec_decode = contracts.SpecDecode.DRAFT_TARGET
        draft_choice: str | None = draft_repo
        why.append(f"draft-target spec-decode: {draft_choice} fits the spare budget")
    else:
        spec_decode = contracts.SpecDecode.NGRAM
        draft_choice = None
        why.append("n-gram spec-decode: no VRAM for a draft model (zero-cost path)")

    backend = contracts.BackendChoice(
        contracts.BackendKind.LLAMACPP,
        contracts.BringUp.NATIVE_WINDOWS,
        "http://127.0.0.1:8080/v1",
    )
    why.append(
        "default native-Windows llama.cpp (CUDA, no WSL2) - "
        "the Windows-tuned local backend"
    )

    target_ctx = _ctx_from_budget(kv_budget, kv_dtype, model)
    est_vram_mb = (
        model.est_weight_mb + _kv_bytes(target_ctx, kv_dtype, model) // _MB + _SAFETY_MB
    )
    est_toks = _decode_estimate(primary_gpu, model, placement)

    if "Laptop" in (primary_gpu.name or ""):
        warnings.append("shared display GPU - expect eviction; enable prefill warming")

    kv_fraction = round(kv_budget / usable_total, 2) if usable_total else 0.0

    if objective is contracts.Objective.QUALITY_FIRST:
        cache_type_k = cache_type_v = "f16"
    elif primary_gpu.supports_int8:
        cache_type_k = cache_type_v = "q8_0"
    else:
        cache_type_k = cache_type_v = "f16"

    return contracts.ExecutionPlan(
        objective=objective,
        model=model,
        placement=placement,
        kv=contracts.KvCacheConfig(
            dtype=kv_dtype,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            enable_block_reuse=True,
            free_gpu_memory_fraction=kv_fraction,
            host_cache_size_bytes=0,
        ),
        levers=contracts.DecodeLevers(
            cuda_graphs=primary_gpu.supports_cuda_graphs,
            spec_decode=spec_decode,
            draft_model=draft_choice,
        ),
        backend=backend,
        target_ctx_tokens=target_ctx,
        est_vram_mb=est_vram_mb,
        est_decode_tok_s=est_toks,
        rationale=tuple(why),
        warnings=tuple(warnings),
    )
