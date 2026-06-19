"""Candidate-model catalog and budget-driven selection helpers (spec.md §1.3).

Static data only — no hardware coupling. Selection helpers take a plain
budget_mb argument; the caller (the PLAN rules, HER-12) derives that budget
from a probed GpuCapabilities (HER-8). Entries reuse contracts.ModelChoice
directly so the catalog *is* the plan's model-choice shape.

"Fully fitting" selection goes through mapper.formulas.layers_that_fit
rather than a weight-footprint shortcut: a model only counts as fully
resident if every one of its layers fits the budget, not just its total
weight size.
"""

from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper import formulas

_MB = 1024 * 1024

# A small, realistic spread of INT4-AWQ-quantized instruction models, sized
# for the Windows + NVIDIA local-fit range this project targets (spec.md §1,
# §6). nvidia/Nemotron-4B-Instruct's footprint matches spec.md §1.4's worked
# example exactly (4.0B params, ~2200 MB, 32 layers).
CATALOG: tuple[contracts.ModelChoice, ...] = (
    contracts.ModelChoice(
        repo="Qwen/Qwen2.5-0.5B-Instruct",
        params_b=0.5,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=300,
        n_layers=24,
    ),
    contracts.ModelChoice(
        repo="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.5,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=900,
        n_layers=28,
    ),
    contracts.ModelChoice(
        repo="meta-llama/Llama-3.2-3B-Instruct",
        params_b=3.2,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=1850,
        n_layers=28,
    ),
    contracts.ModelChoice(
        repo="microsoft/Phi-3.5-mini-instruct",
        params_b=3.8,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=2100,
        n_layers=32,
    ),
    contracts.ModelChoice(
        repo="nvidia/Nemotron-4B-Instruct",
        params_b=4.0,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=2200,
        n_layers=32,
    ),
    contracts.ModelChoice(
        repo="mistralai/Mistral-7B-Instruct-v0.3",
        params_b=7.3,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=4100,
        n_layers=32,
    ),
    contracts.ModelChoice(
        repo="Qwen/Qwen2.5-7B-Instruct",
        params_b=7.6,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=4300,
        n_layers=28,
    ),
    contracts.ModelChoice(
        repo="meta-llama/Llama-3.1-8B-Instruct",
        params_b=8.0,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=4500,
        n_layers=32,
    ),
    contracts.ModelChoice(
        repo="google/Gemma-2-9b-it",
        params_b=9.24,
        weight_quant=contracts.WeightQuant.INT4_AWQ,
        est_weight_mb=5200,
        n_layers=42,
    ),
)


def _candidates(weight_quant: contracts.WeightQuant) -> list[contracts.ModelChoice]:
    return [m for m in CATALOG if m.weight_quant is weight_quant]


def _weight_bytes_per_layer(model: contracts.ModelChoice) -> float:
    return (model.est_weight_mb * _MB) / model.n_layers


def largest_fitting(
    weight_quant: contracts.WeightQuant, budget_mb: int
) -> contracts.ModelChoice:
    """Largest model whose total weight footprint fits budget_mb.

    A footprint check only (spec.md §1.3 Rule 3) — does not guarantee every
    layer is GPU-resident; use largest_fully_fitting for that guarantee.
    """
    fitting = [m for m in _candidates(weight_quant) if m.est_weight_mb <= budget_mb]
    if not fitting:
        raise ValueError(f"no {weight_quant.value} model fits a {budget_mb} MB budget")
    return max(fitting, key=lambda m: m.est_weight_mb)


def largest_fully_fitting(
    weight_quant: contracts.WeightQuant, budget_mb: int
) -> contracts.ModelChoice:
    """Largest model where ALL layers fit resident in budget_mb.

    Enforced via formulas.layers_that_fit: a model only qualifies if the
    number of layers the budget can hold equals its total layer count, so
    "fully fitting" never silently picks a model that needs CPU offload.
    """
    budget_bytes = budget_mb * _MB
    fully_resident = [
        m
        for m in _candidates(weight_quant)
        if formulas.layers_that_fit(
            _weight_bytes_per_layer(m), budget_bytes, m.n_layers
        )
        == m.n_layers
    ]
    if not fully_resident:
        raise ValueError(
            f"no {weight_quant.value} model is fully resident in a "
            f"{budget_mb} MB budget"
        )
    return max(fully_resident, key=lambda m: m.est_weight_mb)


def draft_for(model: contracts.ModelChoice) -> contracts.ModelChoice:
    """The smallest same-quant catalog model (other than `model`) suitable
    as a speculative-decode draft for `model` (spec.md §4.3 draft-target).
    """
    candidates = [m for m in _candidates(model.weight_quant) if m.repo != model.repo]
    if not candidates:
        raise ValueError(f"no draft candidate available for {model.repo}")
    return min(candidates, key=lambda m: m.est_weight_mb)
