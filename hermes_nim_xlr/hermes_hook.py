"""Hermes transport hook — auto-registers XLRTransport for chat_completions mode.

Installed by ``scripts/install-xlr.ps1`` as a ``.pth`` file in the Hermes
venv's site-packages.  When Hermes starts, this module is imported
automatically, detects the GPU, generates an execution plan, and registers
``XLRTransport`` as the ``chat_completions`` transport — so every API request
carries plan-derived config (``cache_prompt``, KV fraction, CUDA graphs,
``n_gpu_layers``, etc.) in ``extra_body``.

If the GPU probe fails, no NVIDIA GPU is found, or ``XLR_DISABLED`` is set,
the hook is a silent no-op — Hermes falls back to its stock
``ChatCompletionsTransport``.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger("hermes_nim_xlr.hook")

_XLR_DISABLED = os.environ.get("XLR_DISABLED", "").lower() in ("1", "true", "yes")

if not _XLR_DISABLED:
    try:
        from agent.transports import register_transport

        from hermes_nim_xlr.mapper import detect, plan
        from hermes_nim_xlr.transport import XLRTransport

        _host = detect()

        if _host.gpus:
            _plan = plan(_host, min_context_tokens=65536)

            class _AutoXLRTransport(XLRTransport):
                """XLRTransport with plan pre-computed at import time."""

                def __init__(self) -> None:
                    super().__init__(
                        execution_plan=_plan,
                        endpoint_url=_plan.backend.serve_endpoint,
                    )

            register_transport("chat_completions", _AutoXLRTransport)

            _logger.debug(
                "XLR hook active: %s | %s",
                _host.gpus[0].name,
                _plan.model.repo,
            )
        else:
            _logger.debug("XLR hook skipped: no NVIDIA GPU detected")
    except Exception:
        pass
