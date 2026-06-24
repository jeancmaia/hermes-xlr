"""Host & GPU probe — the DETECT phase.

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
import shutil
import subprocess
import sys

from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper._bandwidths import _lookup_bandwidth

# pynvml is optional — imported lazily in _probe_gpu_pynvml so the module
# stays importable without it.
try:
    import pynvml as _pynvml  # type: ignore[import-untyped]
except ImportError:
    _pynvml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------#
# Arch detection
# ---------------------------------------------------------------------------#


def _compute_capability_to_arch(major: int, minor: int) -> contracts.GpuArch:
    """Map a CUDA compute capability (major, minor) → GpuArch."""
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


# ---------------------------------------------------------------------------#
# Bandwidth / PCIe helpers
# ---------------------------------------------------------------------------#


def _smi_query(fields: list[str]) -> list[dict[str, str]] | None:
    """Run ``nvidia-smi --query-gpu=...`` and parse CSV per-GPU rows.

    The ``fields`` parameter must be ordered as:
    [index, name, memory.total, memory.free, compute_cap, driver_version]
    """
    csv_fields = ",".join(fields)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={csv_fields}",
                "--format=csv,noheader",
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
        if len(parts) >= 5:
            try:
                cc_major, cc_minor = parts[4].split(".", 1)
            except (ValueError, TypeError, IndexError):
                cc_major, cc_minor = "0", "0"
            gpu_data: dict[str, str] = {
                "index": parts[0],
                "name": parts[1],
                "memory.total": parts[2].replace(" MiB", ""),
                "memory.free": parts[3].replace(" MiB", ""),
                "compute_cap_major": cc_major,
                "compute_cap_minor": cc_minor,
            }
            if len(parts) > 5:
                gpu_data["driver_version"] = parts[5]
            gpus.append(gpu_data)
    return gpus if gpus else None


# ---------------------------------------------------------------------------#
# Probes
# ---------------------------------------------------------------------------#


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
            name = nvml.nvmlDeviceGetName(handle).decode("utf-8", errors="replace")
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
    gpus = _smi_query(
        [
            "index",
            "name",
            "memory.total",
            "memory.free",
            "compute_cap",
            "driver_version",
        ]
    )
    if gpus is None:
        return None

    result: list[dict] = []
    for gpu in gpus:
        try:
            cc_major = int(gpu.get("compute_cap_major", "0"))
            cc_minor = int(gpu.get("compute_cap_minor", "0"))
            index = int(gpu.get("index", 0))
            vram_total_mb = int(gpu.get("memory.total", "0"))
            vram_free_mb = int(gpu.get("memory.free", "0"))
        except (ValueError, TypeError):
            cc_major, cc_minor = 0, 0
            index = 0
            vram_total_mb = 0
            vram_free_mb = 0
        result.append(
            {
                "index": index,
                "name": gpu.get("name", ""),
                "vram_total_mb": vram_total_mb,
                "vram_free_mb": vram_free_mb,
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
        supports_fp8=arch
        in {
            contracts.GpuArch.ADA,
            contracts.GpuArch.HOPPER,
            contracts.GpuArch.BLACKWELL,
        },
        supports_int8=cc_major > 7 or (cc_major == 7 and cc_minor >= 5),
        supports_cuda_graphs=cc_major >= 7,
        driver_version=raw.get("driver_version", ""),
    )


def _detect_os_and_wsl() -> tuple[str, bool]:
    """Detect OS and whether running inside WSL."""
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
                # wmic output: "Capacity\n<bytes>\n..."; header skipped by isdigit()
                total_bytes = sum(
                    int(line.strip())
                    for line in result.stdout.strip().splitlines()
                    if line.strip().isdigit()
                )
                return round(total_bytes / (1024**3), 1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: try Get-WmiObject via PowerShell
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return round(int(result.stdout.strip()) / (1024**3), 1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return 0.0

    # macOS
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            mem = ctypes.c_int64()
            size = ctypes.c_size_t(ctypes.sizeof(mem))
            libc.sysctlbyname(
                b"hw.memsize", ctypes.byref(mem), ctypes.byref(size), None, 0
            )
            return round(mem.value / (1024**3), 1)
        except Exception:  # noqa: BLE001
            pass

    # Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except OSError:
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
    # Prefer Docker over Podman when both are installed.
    has_toolkit = shutil.which("nvidia-container-toolkit") is not None
    if not has_toolkit:
        # Some installs only leave the runtime spec
        has_toolkit = os.path.exists(
            "/usr/share/nvidia-container-toolkit"
        ) or os.path.exists("/etc/nvidia-container-runtime")
    return runtime, has_toolkit


# ---------------------------------------------------------------------------#
# Public API
# ---------------------------------------------------------------------------#


def detect() -> contracts.HostCapabilities:
    """Probe the host and return its capabilities.

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
