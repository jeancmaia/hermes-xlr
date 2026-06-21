"""Tests for the model catalog and budget-driven selection helpers (HER-9).

Catalog: src/hermes_nim_xlr/mapper/catalog.py — entries grounded in spec.md
§1.4's worked example (nvidia/Nemotron-4B-Instruct: 4.0B params, ~2200 MB,
32 layers, INT4_AWQ), plus a wider spread of local-friendly open-weight
families (Phi, Mistral, Gemma) at 4-9B.
"""

import pytest

from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper import catalog, formulas

QUANT = contracts.WeightQuant.INT4_AWQ

# ---------------------------------------------------------------------------
# largest_fitting — a footprint-only check across a range of budgets.
# ---------------------------------------------------------------------------


def test_largest_fitting_picks_nemotron_at_its_own_footprint():
    model = catalog.largest_fitting(QUANT, budget_mb=2300)
    assert model.repo == "nvidia/Nemotron-4B-Instruct"


def test_largest_fitting_picks_largest_model_with_room():
    model = catalog.largest_fitting(QUANT, budget_mb=5000)
    assert model.repo == "meta-llama/Llama-3.1-8B-Instruct"


def test_largest_fitting_picks_overall_largest_catalog_entry():
    model = catalog.largest_fitting(QUANT, budget_mb=5300)
    assert model.repo == "google/Gemma-2-9b-it"


def test_largest_fitting_picks_smallest_model_at_tight_budget():
    model = catalog.largest_fitting(QUANT, budget_mb=350)
    assert model.repo == "Qwen/Qwen2.5-0.5B-Instruct"


def test_largest_fitting_raises_when_nothing_fits():
    with pytest.raises(ValueError):
        catalog.largest_fitting(QUANT, budget_mb=200)


def test_largest_fitting_raises_for_unrepresented_quant():
    with pytest.raises(ValueError):
        catalog.largest_fitting(contracts.WeightQuant.FP16, budget_mb=100_000)


# ---------------------------------------------------------------------------
# largest_fully_fitting — must go through formulas.layers_that_fit, never a
# weight-footprint shortcut, and must never pick a model needing offload.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budget_mb,expected_repo",
    [
        (350, "Qwen/Qwen2.5-0.5B-Instruct"),
        (2300, "nvidia/Nemotron-4B-Instruct"),
        (5000, "meta-llama/Llama-3.1-8B-Instruct"),
    ],
)
def test_largest_fully_fitting_across_budgets(budget_mb, expected_repo):
    model = catalog.largest_fully_fitting(QUANT, budget_mb)
    assert model.repo == expected_repo


def test_largest_fully_fitting_never_requires_offload():
    for budget_mb in (350, 900, 1850, 2100, 2300, 4100, 4300, 5000, 5300):
        model = catalog.largest_fully_fitting(QUANT, budget_mb)
        fit = formulas.layers_that_fit(
            weight_bytes_per_layer=(model.est_weight_mb * 1024 * 1024) / model.n_layers,
            budget_bytes=budget_mb * 1024 * 1024,
            total_layers=model.n_layers,
        )
        assert fit == model.n_layers, (
            "fully-fitting pick must use every layer GPU-resident"
        )


def test_largest_fully_fitting_raises_when_nothing_is_fully_resident():
    with pytest.raises(ValueError):
        catalog.largest_fully_fitting(QUANT, budget_mb=200)


# ---------------------------------------------------------------------------
# draft_for — speculative-decode draft-model selection (spec.md §4.3).
# ---------------------------------------------------------------------------


def test_draft_for_picks_smallest_other_repo():
    nemotron = next(
        m for m in catalog.CATALOG if m.repo == "nvidia/Nemotron-4B-Instruct"
    )
    draft_repo = catalog.draft_for(nemotron)
    assert draft_repo == "Qwen/Qwen2.5-0.5B-Instruct"


def test_draft_for_never_returns_the_model_itself():
    # The smallest catalog entry still has other same-quant candidates to
    # draft from (it just can't draft for itself).
    smallest = min(catalog.CATALOG, key=lambda m: m.est_weight_mb)
    draft_repo = catalog.draft_for(smallest)
    assert draft_repo != smallest.repo


# ---------------------------------------------------------------------------
# Catalog data integrity — data-only, no hardware coupling.
# ---------------------------------------------------------------------------


def test_catalog_entries_are_plain_model_choices():
    for entry in catalog.CATALOG:
        assert isinstance(entry, contracts.ModelChoice)
        assert entry.est_weight_mb > 0
        assert entry.n_layers > 0
        assert entry.kv_heads > 0
        assert entry.head_dim > 0
        assert entry.max_context_tokens > 0
