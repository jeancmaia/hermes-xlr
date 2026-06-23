"""LlamaCppBackend — native-Windows llama.cpp engine backend.

Manages the ``llama-server.exe`` process lifecycle behind the
EngineBackend ABC. Spawns, health-polls, and gracefully stops the
process; enforces the version-match release gate on startup.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from hermes_nim_xlr.backends.engine import EngineBackend
from hermes_nim_xlr.backends.release_gate import assert_engine_checkpoint_match

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8080
_POLL_INTERVAL = 0.5
_START_TIMEOUT = 30.0
_STOP_TIMEOUT = 10.0


class LlamaCppBackend(EngineBackend):
    """Manages a local ``llama-server.exe`` process.

    Parameters
    ----------
    binary_path:
        Path to ``llama-server.exe``.
    model_path:
        Path to the GGUF model file.
    host:
        Bind address (default 127.0.0.1).
    port:
        Bind port (default 8080).
    n_gpu_layers:
        Number of layers to offload to the GPU. ``-1`` (default) means all
        layers — a fully GPU-resident model.
    engine_version:
        Version string of the engine build (for the release gate).
    checkpoint_toolchain_version:
        Version string of the toolchain that produced the checkpoint
        (for the release gate).
    extra_args:
        Additional CLI arguments forwarded verbatim to ``llama-server``.
    start_timeout:
        Seconds to wait for the engine to become healthy (default 30).
    poll_interval:
        Seconds between health-check polls (default 0.5).
    """

    def __init__(
        self,
        *,
        binary_path: str,
        model_path: str,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        n_gpu_layers: int = -1,
        engine_version: str = "",
        checkpoint_toolchain_version: str = "",
        extra_args: tuple[str, ...] = (),
        start_timeout: float = _START_TIMEOUT,
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self._binary_path = binary_path
        self._model_path = model_path
        self._host = host
        self._port = port
        self._n_gpu_layers = n_gpu_layers
        self._engine_version = engine_version
        self._checkpoint_toolchain_version = checkpoint_toolchain_version
        self._extra_args = extra_args
        self._start_timeout = start_timeout
        self._poll_interval = poll_interval

        self._process: subprocess.Popen[str] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def serve_endpoint(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    @property
    def engine_info(self) -> dict[str, Any]:
        return {
            "version": self._engine_version or "unknown",
            "build": self._build_metadata(),
            "cuda": self._cuda_info(),
            "backend": "llama_cpp",
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._assert_binary_exists()
        self._assert_version_match()

        cmd = self._build_command()
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
            text=True,
        )

        if not self._wait_until_healthy():
            self._cleanup_process()
            raise RuntimeError(
                f"llama-server did not become healthy within "
                f"{self._start_timeout}s on {self.serve_endpoint}"
            )

    def stop(self) -> None:
        proc = self._process
        if proc is None:
            return
        self._process = None

        proc.terminate()
        try:
            proc.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def health(self) -> bool:
        url = f"{self.serve_endpoint}/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_binary_exists(self) -> None:
        if not os.path.isfile(self._binary_path):
            raise FileNotFoundError(
                f"llama-server binary not found: {self._binary_path}"
            )

    def _assert_version_match(self) -> None:
        if self._engine_version or self._checkpoint_toolchain_version:
            assert_engine_checkpoint_match(
                self._engine_version, self._checkpoint_toolchain_version
            )

    def _build_command(self) -> list[str]:
        return [
            self._binary_path,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--model",
            self._model_path,
            "--n-gpu-layers",
            str(self._n_gpu_layers),
            *list(self._extra_args),
        ]

    def _wait_until_healthy(self) -> bool:
        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            if self.health():
                return True
            time.sleep(self._poll_interval)
        return False

    def _cleanup_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        self._process = None
        proc.kill()
        proc.wait()

    def _build_metadata(self) -> dict[str, str]:
        return {"commit": "unknown", "compile_flags": "unknown"}

    def _cuda_info(self) -> dict[str, Any]:
        return {"available": False, "device_count": 0}
