"""EngineBackend ABC — the pluggable inference-engine seam.

Every concrete backend (llama.cpp, TensorRT-LLM, MLX) implements
this interface. Start/stop/health govern lifecycle; serve_endpoint
and engine_info are the introspection contract the transport and
harness rely on.
"""

import abc
from typing import Any


class EngineBackend(abc.ABC):
    """Abstract interface for a pluggable inference-engine backend.

    Subclasses manage an engine *process* (spawn, health-poll, graceful
    stop) and expose its OpenAI-compatible endpoint URL and metadata.
    """

    @abc.abstractmethod
    def start(self) -> None:
        """Start the engine and wait until it is ready to serve.

        Raises:
            RuntimeError: if the binary is not found, the endpoint
                does not become healthy within the timeout, or the
                version-match gate fails.
        """
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        """Gracefully shut down the engine process.

        Implementations should send a graceful shutdown signal, wait
        for the process to exit, and forcibly terminate if the process
        does not exit within a reasonable timeout. It must be safe to
        call stop() even if start() was never called or failed.
        """
        ...

    @abc.abstractmethod
    def health(self) -> bool:
        """Return True if the engine is alive and serving.

        This is a lightweight synchronous check (e.g. HEAD /v1/models
        or a dedicated /health endpoint). It must not block for more
        than a few seconds.
        """
        ...

    @property
    @abc.abstractmethod
    def serve_endpoint(self) -> str:
        """The base URL of the engine's OpenAI-compatible endpoint.

        Example: ``"http://127.0.0.1:8080/v1"``
        """
        ...

    @property
    @abc.abstractmethod
    def engine_info(self) -> dict[str, Any]:
        """Engine metadata.

        Returns:
            dict with at least the keys:
            - ``"version"``:  engine version string
            - ``"build"``:    build metadata (commit, compile flags)
            - ``"cuda"``:     CUDA availability / device count
            - ``"backend"``:  backend kind name
        """
        ...
