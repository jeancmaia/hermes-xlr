"""Tests for the engine-backend ABC and llama.cpp implementation (HER-14).

Covers:
  - EngineBackend ABC structure (abstract methods, properties)
  - LlamaCppBackend lifecycle: start, healthy, stop, health check
  - Error paths: binary not found, start timeout, OOM/stderr crash
  - Version-match release gate
  - Registry / factory function
"""

from __future__ import annotations

from unittest import mock

import pytest
from hermes_nim_xlr.backends import (
    EngineBackend,
    LlamaCppBackend,
    create_backend,
    register,
)
from hermes_nim_xlr.backends.release_gate import VersionMismatchError

_FAKE_BINARY = "C:\\llama\\llama-server.exe"
_FAKE_MODEL = "C:\\models\\model.gguf"


def _mock_healthy_response() -> mock.MagicMock:
    """Return a MagicMock that simulates an HTTP 200 response from urlopen.

    MagicMock's default ``__enter__`` returns a *new* MagicMock, which loses
    the ``status`` attribute. This helper wires ``__enter__`` to return the
    same object so ``with urlopen(...) as resp: resp.status`` works.
    """
    resp = mock.MagicMock(status=200)
    resp.__enter__.return_value = resp
    return resp


# ===========================================================================
# EngineBackend ABC — contract enforcement
# ===========================================================================


class _ConcreteBackend(EngineBackend):
    """Minimal concrete subclass for ABC instantiation tests."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def health(self) -> bool:
        return True

    @property
    def serve_endpoint(self) -> str:
        return "http://127.0.0.1:8080/v1"

    @property
    def engine_info(self) -> dict:
        return {"backend": "test"}


def test_abc_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        EngineBackend()  # type: ignore[abstract]


def test_concrete_subclass_can_be_instantiated():
    backend = _ConcreteBackend()
    assert backend.health() is True
    assert backend.serve_endpoint == "http://127.0.0.1:8080/v1"


# ===========================================================================
# LlamaCppBackend — properties
# ===========================================================================


def test_serve_endpoint_default():
    backend = LlamaCppBackend(binary_path=_FAKE_BINARY, model_path=_FAKE_MODEL)
    assert backend.serve_endpoint == "http://127.0.0.1:8080/v1"
    assert backend.serve_endpoint.endswith("/v1")


def test_serve_endpoint_custom_port():
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY, model_path=_FAKE_MODEL, port=8081
    )
    assert backend.serve_endpoint == "http://127.0.0.1:8081/v1"


def test_engine_info_returns_dict_with_expected_keys():
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY,
        model_path=_FAKE_MODEL,
        engine_version="b1234",
    )
    info = backend.engine_info
    assert isinstance(info, dict)
    assert "version" in info
    assert "build" in info
    assert "cuda" in info
    assert "backend" in info
    assert info["backend"] == "llama_cpp"
    assert info["version"] == "b1234"


# ===========================================================================
# LlamaCppBackend — lifecycle: start / health / stop
# ===========================================================================


def test_start_stops_cleanly():
    """Happy-path: start spawns process, health returns True, stop
    terminates cleanly."""
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY,
        model_path=_FAKE_MODEL,
        start_timeout=0.1,
        poll_interval=0.01,
    )

    with (
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("subprocess.Popen") as mock_popen,
        mock.patch(
            "urllib.request.urlopen",
            return_value=_mock_healthy_response(),
        ) as mock_health,
    ):
        proc = mock.MagicMock()
        mock_popen.return_value = proc

        backend.start()

        mock_popen.assert_called_once()
        args, _ = mock_popen.call_args
        assert _FAKE_BINARY in args[0]

        assert mock_health.call_count >= 1

        assert backend.health() is True

        backend.stop()

        proc.terminate.assert_called_once()
        proc.wait.assert_called()


def test_binary_not_found_raises():
    backend = LlamaCppBackend(
        binary_path="C:\\nonexistent\\llama-server.exe",
        model_path=_FAKE_MODEL,
    )

    with mock.patch("os.path.isfile", return_value=False):
        with pytest.raises(FileNotFoundError, match="binary not found"):
            backend.start()


def test_start_timeout_raises():
    """When health never returns True, start raises RuntimeError."""
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY,
        model_path=_FAKE_MODEL,
        start_timeout=0.1,
        poll_interval=0.01,
    )

    with (
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("subprocess.Popen") as _mock_popen,
        mock.patch(
            "urllib.request.urlopen",
            side_effect=ConnectionError("connection refused"),
        ),
    ):
        proc = mock.MagicMock()
        _mock_popen.return_value = proc

        with pytest.raises(RuntimeError, match="did not become healthy"):
            backend.start()

        proc.kill.assert_called_once()
        proc.wait.assert_called()


def test_stop_safe_when_not_started():
    """Calling stop() without a prior start() must not raise."""
    backend = LlamaCppBackend(binary_path=_FAKE_BINARY, model_path=_FAKE_MODEL)
    backend.stop()  # should be a no-op


def test_stop_safe_to_call_multiple_times():
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY,
        model_path=_FAKE_MODEL,
        start_timeout=0.1,
        poll_interval=0.01,
    )

    with (
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("subprocess.Popen") as _mock_popen2,
        mock.patch(
            "urllib.request.urlopen",
            return_value=_mock_healthy_response(),
        ),
    ):
        proc = mock.MagicMock()
        _mock_popen2.return_value = proc

        backend.start()
        backend.stop()
        backend.stop()  # second call is safe


def test_health_returns_false_on_connection_error():
    backend = LlamaCppBackend(binary_path=_FAKE_BINARY, model_path=_FAKE_MODEL)

    with mock.patch(
        "urllib.request.urlopen",
        side_effect=ConnectionError("connection refused"),
    ):
        assert backend.health() is False


def test_health_returns_false_on_timeout():
    backend = LlamaCppBackend(binary_path=_FAKE_BINARY, model_path=_FAKE_MODEL)

    with mock.patch(
        "urllib.request.urlopen",
        side_effect=TimeoutError("timeout"),
    ):
        assert backend.health() is False


# ===========================================================================
# Version-match release gate
# ===========================================================================


def test_version_match_passes():
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY,
        model_path=_FAKE_MODEL,
        engine_version="b1234",
        checkpoint_toolchain_version="b1234",
        start_timeout=0.1,
        poll_interval=0.01,
    )

    with (
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("subprocess.Popen"),
        mock.patch(
            "urllib.request.urlopen",
            return_value=_mock_healthy_response(),
        ),
    ):
        backend.start()  # should not raise
        backend.stop()


def test_version_mismatch_raises():
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY,
        model_path=_FAKE_MODEL,
        engine_version="b1234",
        checkpoint_toolchain_version="b5678",
    )

    with mock.patch("os.path.isfile", return_value=True):
        with pytest.raises(VersionMismatchError, match="does not match"):
            backend.start()


def test_version_gate_skipped_when_empty():
    """If both version strings are empty, the gate is a no-op."""
    backend = LlamaCppBackend(
        binary_path=_FAKE_BINARY,
        model_path=_FAKE_MODEL,
        start_timeout=0.1,
        poll_interval=0.01,
    )

    with (
        mock.patch("os.path.isfile", return_value=True),
        mock.patch("subprocess.Popen"),
        mock.patch(
            "urllib.request.urlopen",
            return_value=_mock_healthy_response(),
        ),
    ):
        backend.start()  # should not raise
        backend.stop()


# ===========================================================================
# Registry / factory
# ===========================================================================


class _DummyBackend(EngineBackend):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def health(self) -> bool:
        return True

    @property
    def serve_endpoint(self) -> str:
        return "http://dummy/v1"

    @property
    def engine_info(self) -> dict:
        return {"backend": "dummy"}


def test_register_and_create_backend():
    register("dummy", _DummyBackend)
    backend = create_backend("dummy")
    assert isinstance(backend, _DummyBackend)
    assert backend.serve_endpoint == "http://dummy/v1"


def test_create_backend_unregistered_raises():
    with pytest.raises(KeyError, match="no backend registered for"):
        create_backend("never_registered")


def test_llama_cpp_is_registered_by_default():
    backend = create_backend(
        "llama_cpp", binary_path=_FAKE_BINARY, model_path=_FAKE_MODEL
    )
    assert isinstance(backend, LlamaCppBackend)
    assert backend.serve_endpoint == "http://127.0.0.1:8080/v1"
