"""Enforces the engine <-> checkpoint version-match release gate.

A quantized checkpoint produced by one toolchain version can silently
misload or decode garbage under a mismatched engine build. This is the
generic gate; later sprints wire it to real engine and checkpoint metadata.
"""


class VersionMismatchError(RuntimeError):
    """Raised when the engine version and checkpoint toolchain version diverge."""


def assert_engine_checkpoint_match(
    engine_version: str, checkpoint_toolchain_version: str
) -> None:
    if engine_version != checkpoint_toolchain_version:
        raise VersionMismatchError(
            f"Engine version {engine_version!r} does not match checkpoint toolchain "
            f"version {checkpoint_toolchain_version!r}; refusing to load weights."
        )
