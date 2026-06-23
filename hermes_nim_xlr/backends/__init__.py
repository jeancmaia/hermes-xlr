"""Pluggable inference-engine backends and their tuned per-backend configs.

Registry
--------
Use ``register()`` to associate a ``BackendKind`` with a concrete
``EngineBackend`` subclass, then ``create_backend()`` to instantiate
the right backend from an ``ExecutionPlan`` or ``BackendChoice``.
"""

from __future__ import annotations

from hermes_nim_xlr.backends.engine import EngineBackend
from hermes_nim_xlr.backends.llama_cpp import LlamaCppBackend

__all__: list[str] = [
    "EngineBackend",
    "LlamaCppBackend",
    "create_backend",
    "register",
]

_REGISTRY: dict[str, type[EngineBackend]] = {}


def register(kind: str, cls: type[EngineBackend]) -> None:
    """Register a concrete backend class for *kind* (e.g. ``"llama_cpp"``)."""
    _REGISTRY[kind] = cls


def create_backend(kind: str, **kwargs: object) -> EngineBackend:
    """Create an engine backend by registered kind name.

    Args:
        kind: Backend kind string (e.g. ``"llama_cpp"``).
        **kwargs: Forwarded to the backend constructor.

    Returns:
        An instantiated ``EngineBackend``.

    Raises:
        KeyError: if *kind* is not registered.
    """
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        available = list(_REGISTRY)
        raise KeyError(f"no backend registered for {kind!r}; available: {available}")
    return cls(**kwargs)


# Register the built-in backends.
register("llama_cpp", LlamaCppBackend)
