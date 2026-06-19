"""Shared pytest fixtures; profile-specific ones live under tests/fixtures/."""

import pytest


@pytest.fixture
def package_name() -> str:
    """Placeholder proving the fixtures wiring; replace as real fixtures land."""
    return "hermes_nim_xlr"
