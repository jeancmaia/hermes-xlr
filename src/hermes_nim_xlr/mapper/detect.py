"""Host & GPU probe — the DETECT phase (spec.md §1.1).

Discovers the host's GPU(s), OS, and runtime environment using:
  - pynvml (``nvidia-ml-py``) as the primary GPU probe — cheap to import,
    no CUDA/torch dependency.
  - ``nvidia-smi`` parsing as a zero-dependency fallback.
  - stdlib OS / process / file probes for host-level detection.

Returns an immutable ``HostCapabilities`` record — the DETECT output that
the PLAN rules (HER-12) consume.

Bandwidth figures are a static lookup by GPU name pattern; NVML does not
expose memory or PCIe bandwidth directly.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys

from hermes_nim_xlr import contracts

# pynvml is optional — imported lazily in _probe_gpu_pynvml so the module
# stays importable without it.  The _HAS_PYNVML sentinel avoids re-trying
# the import every call.
try:
    import pynvml as _pynvml  # type: ignore[import-untyped]

    _HAS_PYNVML = True
except ImportError:
    _pynvml = None  # type: ignore[assignment]
    _HAS_PYNVML = False


# ---------------------------------------------------------------------------
# Static bandwidth lookup table
#
# Memory bandwidth (GB/s) and typical PCIe generation / lane count for common
# NVIDIA GPUs.  Keyed by substrings matched against the GPU name string.
# Unknown GPUs produce None for bandwidths.
# ---------------------------------------------------------------------------

_BANDWIDTH_ENTRY = tuple[float | None, int | None, int | None]
"""A (mem_bandwidth_gbs, pcie_gen, pcie_lanes) row."""

_BANDWIDTH_LOOKUP: dict[str, list[_BANDWIDTH_ENTRY]] = {
    # ---- GeForce RTX 30-series (Ampere) ----
    "RTX 3050": [(224.0, 4, 8)],
    "RTX 3050 Laptop": [(170.0, 4, 8)],
    "RTX 3060": [(360.0, 4, 16)],
    "RTX 3060 Laptop": [(192.0, 4, 8)],
    "RTX 3060 Ti": [(448.0, 4, 16)],
    "RTX 3070": [(448.0, 4, 16)],
    "RTX 3070 Laptop": [(384.0, 4, 8)],
    "RTX 3070 Ti": [(448.0, 4, 16)],
    "RTX 3080": [(760.0, 4, 16)],
    "RTX 3080 Laptop": [(384.0, 4, 8)],
    "RTX 3080 Ti": [(912.0, 4, 16)],
    "RTX 3090": [(936.0, 4, 16)],
    "RTX 3090 Ti": [(1008.0, 4, 16)],
    # ---- GeForce RTX 40-series (Ada Lovelace) ----
    "RTX 4050 Laptop": [(192.0, 4, 8)],
    "RTX 4060": [(272.0, 4, 8)],
    "RTX 4060 Laptop": [(256.0, 4, 8)],
    "RTX 4060 Ti": [(288.0, 4, 8)],
    "RTX 4070": [(504.0, 4, 16)],
    "RTX 4070 Laptop": [(256.0, 4, 8)],
    "RTX 4070 Ti": [(576.0, 4, 16)],
    "RTX 4070 Ti Super": [(672.0, 4, 16)],
    "RTX 4080": [(736.0, 4, 16)],
    "RTX 4080 Laptop": [(512.0, 4, 8)],
    "RTX 4080 Super": [(736.0, 4, 16)],
    "RTX 4090": [(1008.0, 4, 16)],
    "RTX 4090 Laptop": [(576.0, 4, 8)],
    # ---- GeForce RTX 50-series (Blackwell) ----
    "RTX 5060": [(512.0, 5, 8)],
    "RTX 5070": [(672.0, 5, 16)],
    "RTX 5080": [(960.0, 5, 16)],
    "RTX 5090": [(1792.0, 5, 16)],
    "RTX 5090 Laptop": [(880.0, 5, 8)],
    # ---- Enterprise / Pro ----
    "A100": [(1555.0, 4, 16)],
    "H100": [(3352.0, 5, 16)],
    "B200": [(4304.0, 5, 16)],
    "T4": [(320.0, 3, 16)],
    "L4": [(300.0, 4, 16)],
    "L40S": [(864.0, 4, 16)],
    "A10": [(600.0, 4, 16)],
    "A30": [(933.0, 4, 16)],
    "A40": [(696.0, 4, 16)],
}

# PCIe per-lane bandwidth in GB/s (bidirectional, 1e9 bytes).
_PCIE_GEN_BW: dict[int, float] = {
    1: 0.250,
    2: 0.500,
    3: 0.985,
    4: 1.969,
    5: 3.938,
    6: 7.563,
}


# ---------------------------------------------------------------------------
# Arch detection
# ---------------------------------------------------------------------------

def _compute_capability_to_arch(major: int, minor: int) -> contracts.GpuArch:
    """Map a CUDA compute capability (major, minor) → GpuArch (spec.md §1.1)."""
    cc = (major, minor)
    if cc >= (10, 0):
        return contracts.GpuArch.BLACKWELL
    if cc >= (9, 0):
        return contracts.GpuArch.HOPPER
    if cc >= (8, 9):
        return contracts.GpuArch.ADA
    if cc >= (8, 0):
        return contracts.GpuArch.AMPERE
    return contracts.GpuArch.OTHER


# ---------------------------------------------------------------------------
# Bandwidth / PCIe helpers
# ---------------------------------------------------------------------------

def _lookup_bandwidth(
    name: str,
) -> tuple[float | None, float | None]:
    """Lookup memory bandwidth (GB/s) and PCIe bandwidth (GB/s) by GPU name.

    Matching uses **word-level token overlap** rather than substring matching,
    so a key ``"RTX 3050 Laptop"`` correctly matches the NVIDIA name
    ``"NVIDIA GeForce RTX 3050 6GB Laptop GPU"`` (every word in the key
    appears in the name as a token, even though the full substring does not).

    The longest key (by character length) whose every word-token appears in
    the GPU name wins — this gives correct greedy behavior for qualified names
    (``"RTX 3080 Ti"`` beats ``"RTX 3080"``).
    """
    name_lower = name.lower()
    # Split on any non-alphanumeric boundary so hyphenated tokens like
    # "A100-SXM4-80GB" produce individual words: {"a100", "sxm4", "80gb"}.
    name_tokens = set(re.split(r"[^a-z0-9]+", name_lower))

    matched_key = ""
    matched_entry: _BANDWIDTH_ENTRY | None = None

    for key, entries in _BANDWIDTH_LOOKUP.items():
        key_tokens = key.lower().split()
        if not key_tokens:
            continue
        # Every word in the key must appear somewhere in the GPU name.
        if all(t in name_tokens for t in key_tokens):
            if len(key) > len(matched_key):
                matched_key = key
                matched_entry = entries[0]

    if matched_entry is None:
        return None, None

    mem_bw, pcie_gen, pcie_lanes = matched_entry
    if pcie_gen is not None and pcie_lanes is not None:
        pcie_bw = _PCIE_GEN_BW.get(pcie_gen, 1.969) * pcie_lanes
    else:
        pcie_bw = None
    return mem_bw, pcie_bw


def _smi_query(fields: list[str]) -> list[dict[str, str]] | None:
    """Run ``nvidia-smi --query-gpu=...`` and parse CSV per-GPU rows."""
    csv_fields = ",".join(fields)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={csv_fields}",
                "--format=csv,noheader,nocpu",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    gpus: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) >= 6:
            gpu_data: dict[str, str] = {
                "index": parts[0],
                "name": parts[1],
                "memory.total": parts[2].replace(" MiB", ""),
                "memory.free": parts[3].replace(" MiB", ""),
                "compute_cap_major": parts[4],
                "compute_cap_minor": parts[5],
            }
            if len(parts) > 6:
                gpu_data["driver_version"] = parts[6]
            gpus.append(gpu_data)
    return gpus if gpus else None


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _probe_gpu_pynvml() -> list[dict] | None:
    """Probe all GPUs via pynvml (primary path).

    Uses the module-level ``_pynvml`` / ``_HAS_PYNVML`` sentinel so callers
    can patch ``detect._pynvml`` in tests without fighting a local import.
    """
    nvml = _pynvml
    if nvml is None:
        return None

    try:
        nvml.nvmlInit()
    except nvml.NVMLError:
        return None

    gpus: list[dict] = []
    try:
        count = nvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode(
                "utf-8", errors="replace"
            )
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            cc_major, cc_minor = (
                nvml.nvmlDeviceGetCudaComputeCapability(handle)
                if hasattr(nvml, "nvmlDeviceGetCudaComputeCapability")
                else (0, 0)
            )
            try:
                driver = nvml.nvmlSystemGetDriverVersion().decode(
                    "utf-8", errors="replace"
                )
            except nvml.NVMLError:
                driver = ""

            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "vram_total_mb": mem.total // (1024 * 1024),
                    "vram_free_mb": mem.free // (1024 * 1024),
                    "compute_cap_major": cc_major,
                    "compute_cap_minor": cc_minor,
                    "driver_version": driver,
                }
            )
    finally:
        try:
            nvml.nvmlShutdown()
        except nvml.NVMLError:
            pass

    return gpus if gpus else None


def _probe_gpu_smi() -> list[dict] | None:
    """Probe all GPUs via ``nvidia-smi`` (fallback path)."""
    gpus = _smi_query(["index", "name", "memory.total", "memory.free",
                         "compute_cap_major", "compute_cap_minor",
                         "driver_version"])
    if gpus is None:
        return None

    result: list[dict] = []
    for gpu in gpus:
        try:
            cc_major = int(gpu.get("compute_cap_major", "0"))
            cc_minor = int(gpu.get("compute_cap_minor", "0"))
        except (ValueError, TypeError):
            cc_major, cc_minor = 0, 0
        result.append(
            {
                "index": int(gpu.get("index", 0)),
                "name": gpu.get("name", ""),
                "vram_total_mb": int(gpu.get("memory.total", "0")),
                "vram_free_mb": int(gpu.get("memory.free", "0")),
                "compute_cap_major": cc_major,
                "compute_cap_minor": cc_minor,
                "driver_version": gpu.get("driver_version", ""),
            }
        )
    return result


def _build_gpu(raw: dict) -> contracts.GpuCapabilities:
    """Translate a raw probe dict into a frozen ``GpuCapabilities`` record."""
    cc_major = raw.get("compute_cap_major", 0)
    cc_minor = raw.get("compute_cap_minor", 0)
    name = raw.get("name", "")
    arch = _compute_capability_to_arch(cc_major, cc_minor)
    mem_bw, pcie_bw = _lookup_bandwidth(name)
    return contracts.GpuCapabilities(
        index=raw.get("index", 0),
        name=name,
        arch=arch,
        compute_capability=(cc_major, cc_minor),
        vram_total_mb=raw.get("vram_total_mb", 0),
        vram_free_mb=raw.get("vram_free_mb", 0),
        mem_bandwidth_gbs=mem_bw,
        pcie_bandwidth_gbs=pcie_bw,
        supports_fp8=arch in {contracts.GpuArch.ADA, contracts.GpuArch.HOPPER,
                               contracts.GpuArch.BLACKWELL},
        supports_int8=cc_major > 7 or (cc_major == 7 and cc_minor >= 5),
        supports_cuda_graphs=cc_major >= 7,
        driver_version=raw.get("driver_version", ""),
    )


def _detect_os_and_wsl() -> tuple[str, bool]:
    """Detect OS and whether running inside WSL (spec.md §1.1)."""
    os_name = platform.system()
    is_wsl = False
    if os_name == "Linux":
        try:
            with open("/proc/version") as f:
                is_wsl = "microsoft" in f.read().lower()
        except OSError:
            pass
    return os_name, is_wsl


def _get_cpu_ram_gb() -> float:
    """Detect total CPU RAM in GB (cross-platform)."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["wmic", "memorychip", "get", "Capacity"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                total_bytes = sum(
                    int(line.strip())
                    for line in result.stdout.strip().splitlines()
                    if line.strip().isdigit()
                )
                return round(total_bytes / (1024 ** 3), 1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: try Get-WmiObject via PowerShell
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return round(int(result.stdout.strip()) / (1024 ** 3), 1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0.0

    # Linux / macOS
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            mem = ctypes.c_int64()
            size = ctypes.c_size_t(ctypes.sizeof(mem))
            libc.sysctlbyname(b"hw.memsize", ctypes.byref(mem), ctypes.byref(size),
                              None, 0)
            return round(mem.value / (1024 ** 3), 1)
        except Exception:  # noqa: BLE001
            pass
    return 0.0


def _detect_container_runtime() -> tuple[str | None, bool]:
    """Detect Docker/Podman and whether the NVIDIA Container Toolkit is present.

    Returns (runtime_name_or_None, has_nvidia_toolkit).
    """
    runtime = None
    if shutil.which("docker"):
        runtime = "docker"
    elif shutil.which("podman"):
        runtime = "podman"
    has_toolkit = shutil.which("nvidia-container-toolkit") is not None
    if not has_toolkit:
        # Some installs only leave the runtime spec
        has_toolkit = (
            os.path.exists("/usr/share/nvidia-container-toolkit")
            or os.path.exists("/etc/nvidia-container-runtime")
        )
    return runtime, has_toolkit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect() -> contracts.HostCapabilities:
    """Probe the host and return its capabilities (spec.md §1.1 DETECT).

    This is the primary entry point for the mapper's DETECT phase.  It
    probes GPUs via pynvml (primary) or ``nvidia-smi`` (fallback), detects
    the OS, WSL status, CPU RAM, and container runtime, and returns an
    immutable ``HostCapabilities`` record.
    """
    gpu_raw_list = _probe_gpu_pynvml() or _probe_gpu_smi() or []
    gpus = tuple(_build_gpu(raw) for raw in gpu_raw_list)

    os_name, is_wsl = _detect_os_and_wsl()
    cpu_ram = _get_cpu_ram_gb()
    runtime, has_toolkit = _detect_container_runtime()

    return contracts.HostCapabilities(
        os=os_name,
        is_wsl=is_wsl,
        cpu_ram_gb=cpu_ram,
        container_runtime=runtime,
        has_nvidia_container_toolkit=has_toolkit,
        gpus=gpus,
    )