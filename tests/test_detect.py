"""Tests for the DETECT — host & GPU probe (HER-8).

Covers:
  - GPU probe: pynvml (primary) and nvidia-smi (fallback) paths
  - Arch detection from compute capability
  - Bandwidth lookup by GPU name
  - Host detection: OS, WSL, CPU RAM, container runtime
  - The top-level ``detect()`` integration via mocked externals
"""

from __future__ import annotations

from unittest import mock

import pytest
from hermes_nim_xlr import contracts
from hermes_nim_xlr.mapper import detect

# ===========================================================================
# Compute capability → GpuArch
# ===========================================================================


@pytest.mark.parametrize(
    "major,minor,expected_arch",
    [
        (8, 6, contracts.GpuArch.AMPERE),  # RTX 3050/3060/3070 laptop
        (8, 0, contracts.GpuArch.AMPERE),  # A100
        (8, 7, contracts.GpuArch.AMPERE),  # RTX 3080/3090 desktop
        (8, 9, contracts.GpuArch.ADA),  # RTX 4090
        (9, 0, contracts.GpuArch.HOPPER),  # H100
        (10, 0, contracts.GpuArch.BLACKWELL),
        (12, 0, contracts.GpuArch.BLACKWELL),
        (7, 5, contracts.GpuArch.OTHER),  # Turing (no dedicated enum)
        (7, 0, contracts.GpuArch.OTHER),  # Volta
        (6, 1, contracts.GpuArch.OTHER),  # Pascal
    ],
)
def test_compute_capability_to_arch(major, minor, expected_arch):
    assert detect._compute_capability_to_arch(major, minor) is expected_arch


# ===========================================================================
# Bandwidth lookup
# ===========================================================================


@pytest.mark.parametrize(
    "name,expect_mem_bw,expect_pcie_bw",
    [
        ("NVIDIA GeForce RTX 3050 6GB Laptop GPU", 170.0, 1.969 * 8),
        ("NVIDIA GeForce RTX 4090", 1008.0, 1.969 * 16),
        ("NVIDIA GeForce RTX 4090 Laptop GPU", 576.0, 1.969 * 8),
        ("NVIDIA A100-SXM4-80GB", 1555.0, 1.969 * 16),
        ("NVIDIA H100 80GB HBM3", 3352.0, 3.938 * 16),
        ("NVIDIA GeForce RTX 3080 Ti", 912.0, 1.969 * 16),  # longest-key match
        ("NVIDIA GeForce RTX 3080", 760.0, 1.969 * 16),  # desktop variant
        ("Unknown GPU Model", None, None),
        ("Intel UHD Graphics", None, None),
    ],
)
def test_lookup_bandwidth(name, expect_mem_bw, expect_pcie_bw):
    mem_bw, pcie_bw = detect._lookup_bandwidth(name)
    assert mem_bw == expect_mem_bw
    assert pcie_bw == expect_pcie_bw


# ===========================================================================
# _probe_gpu_pynvml — primary path
# ===========================================================================


def test_probe_gpu_pynvml_single_gpu():
    """Simulate a single GPU detected via pynvml (e.g. RTX 3050 laptop)."""
    mock_nvml = mock.MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    handle = mock.MagicMock()
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 3050 6GB Laptop GPU"
    mem_info = mock.MagicMock()
    mem_info.total = 6 * 1024 * 1024 * 1024  # 6 GB
    mem_info.free = 5 * 1024 * 1024 * 1024  # 5 GB
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 6)
    mock_nvml.nvmlSystemGetDriverVersion.return_value = b"555.99"
    # hasattr(mock_nvml, "nvmlDeviceGetCudaComputeCapability") returns True
    # for MagicMock because the attribute will be created on getattr, so hasattr
    # also returns True

    with mock.patch.object(detect, "_pynvml", mock_nvml):
        gpus = detect._probe_gpu_pynvml()

    assert gpus is not None
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
    assert gpus[0]["vram_total_mb"] == 6144  # 6 GB in MiB
    assert gpus[0]["vram_free_mb"] == 5120
    assert gpus[0]["compute_cap_major"] == 8
    assert gpus[0]["compute_cap_minor"] == 6
    assert gpus[0]["driver_version"] == "555.99"


def test_probe_gpu_pynvml_multi_gpu():
    mock_nvml = mock.MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 2
    handle0 = mock.MagicMock()
    handle1 = mock.MagicMock()
    mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = [handle0, handle1]
    mock_nvml.nvmlDeviceGetName.side_effect = [
        b"NVIDIA A100-SXM4-80GB",
        b"NVIDIA A100-SXM4-80GB",
    ]
    mem_info = mock.MagicMock()
    mem_info.total = 80 * 1024 * 1024 * 1024
    mem_info.free = 75 * 1024 * 1024 * 1024
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    mock_nvml.nvmlDeviceGetCudaComputeCapability.return_value = (8, 0)
    mock_nvml.nvmlSystemGetDriverVersion.return_value = b"550.54"
    # hasattr(mock_nvml, "nvmlDeviceGetCudaComputeCapability") returns True
    # for MagicMock because the attribute will be created on getattr, so hasattr
    # also returns True

    with mock.patch.object(detect, "_pynvml", mock_nvml):
        gpus = detect._probe_gpu_pynvml()

    assert gpus is not None
    assert len(gpus) == 2
    assert gpus[0]["index"] == 0
    assert gpus[1]["index"] == 1
    assert gpus[0]["compute_cap_major"] == 8
    assert gpus[0]["driver_version"] == "550.54"


def test_probe_gpu_pynvml_no_module():
    """When _pynvml is None (not installed), return None."""
    with mock.patch.object(detect, "_pynvml", None):
        assert detect._probe_gpu_pynvml() is None


def test_probe_gpu_pynvml_nvml_init_fails():
    class NVMLError(Exception):
        pass

    mock_nvml = mock.MagicMock()
    mock_nvml.NVMLError = NVMLError
    mock_nvml.nvmlInit.side_effect = NVMLError("NVML_ERROR_UNKNOWN")

    with mock.patch.object(detect, "_pynvml", mock_nvml):
        assert detect._probe_gpu_pynvml() is None


# ===========================================================================
# _probe_gpu_smi — fallback path
# ===========================================================================


def test_probe_gpu_smi_parses_csv():
    """Simulate nvidia-smi CSV output for a single RTX 3050."""
    stdout = (
        "0, NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB, 5120 MiB, 8, 6, 555.99\n"
    )
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = stdout
        gpus = detect._probe_gpu_smi()

    assert gpus is not None
    assert len(gpus) == 1
    assert gpus[0]["name"] == "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
    assert gpus[0]["vram_total_mb"] == 6144
    assert gpus[0]["vram_free_mb"] == 5120
    assert gpus[0]["compute_cap_major"] == 8
    assert gpus[0]["compute_cap_minor"] == 6
    assert gpus[0]["driver_version"] == "555.99"


def test_probe_gpu_smi_multiple():
    """Two GPUs are parsed as separate entries."""
    stdout = (
        "0, NVIDIA A100-SXM4-80GB, 81920 MiB, 76800 MiB, 8, 0, 550.54\n"
        "1, NVIDIA A100-SXM4-80GB, 81920 MiB, 76800 MiB, 8, 0, 550.54\n"
    )
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = stdout
        gpus = detect._probe_gpu_smi()

    assert gpus is not None
    assert len(gpus) == 2
    assert all(g["index"] == i for i, g in enumerate(gpus))


def test_probe_gpu_smi_empty():
    """No GPUs → None."""
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        assert detect._probe_gpu_smi() is None


def test_probe_gpu_smi_not_found():
    """nvidia-smi not on PATH → None."""
    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect._probe_gpu_smi() is None


# ===========================================================================
# _build_gpu — raw dict → GpuCapabilities
# ===========================================================================


def test_build_gpu_rtx_3050():
    raw = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "vram_total_mb": 6144,
        "vram_free_mb": 5120,
        "compute_cap_major": 8,
        "compute_cap_minor": 6,
        "driver_version": "555.99",
    }
    gpu = detect._build_gpu(raw)
    assert gpu.index == 0
    assert gpu.arch is contracts.GpuArch.AMPERE
    assert gpu.compute_capability == (8, 6)
    assert gpu.vram_total_mb == 6144
    assert gpu.mem_bandwidth_gbs == 170.0
    assert gpu.supports_fp8 is False
    assert gpu.supports_int8 is True
    assert gpu.supports_cuda_graphs is True


def test_build_gpu_rtx_4090():
    raw = {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090",
        "vram_total_mb": 24576,
        "vram_free_mb": 22000,
        "compute_cap_major": 8,
        "compute_cap_minor": 9,
        "driver_version": "555.99",
    }
    gpu = detect._build_gpu(raw)
    assert gpu.arch is contracts.GpuArch.ADA
    assert gpu.compute_capability == (8, 9)
    assert gpu.mem_bandwidth_gbs == 1008.0
    assert gpu.supports_fp8 is True


def test_build_gpu_unknown():
    raw = {
        "index": 0,
        "name": "Unknown GPU",
        "vram_total_mb": 4096,
        "vram_free_mb": 3000,
        "compute_cap_major": 0,
        "compute_cap_minor": 0,
        "driver_version": "",
    }
    gpu = detect._build_gpu(raw)
    assert gpu.arch is contracts.GpuArch.OTHER
    assert gpu.mem_bandwidth_gbs is None
    assert gpu.supports_fp8 is False
    assert gpu.supports_int8 is False
    assert gpu.supports_cuda_graphs is False


# ===========================================================================
# Host detection
# ===========================================================================


def test_detect_os_and_wsl_native_linux():
    with mock.patch("platform.system", return_value="Linux"):
        read_data = "Linux version 6.2.0"
        with mock.patch("builtins.open", mock.mock_open(read_data=read_data)):
            os_name, is_wsl = detect._detect_os_and_wsl()
    assert os_name == "Linux"
    assert is_wsl is False


def test_detect_os_and_wsl_under_wsl():
    with mock.patch("platform.system", return_value="Linux"):
        read_data = "Linux version 5.15.153.1-microsoft-standard-WSL2"
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data=read_data),
        ):
            os_name, is_wsl = detect._detect_os_and_wsl()
    assert os_name == "Linux"
    assert is_wsl is True


def test_detect_os_and_wsl_windows():
    with mock.patch("platform.system", return_value="Windows"):
        os_name, is_wsl = detect._detect_os_and_wsl()
    assert os_name == "Windows"
    assert is_wsl is False


@mock.patch("os.path.exists", return_value=False)
def test_detect_container_runtime_docker(mock_exists):
    with mock.patch(
        "shutil.which",
        side_effect=lambda x: "/usr/bin/docker" if x == "docker" else None,
    ):
        runtime, has_toolkit = detect._detect_container_runtime()
    assert runtime == "docker"
    # No toolkit path found
    assert has_toolkit is False


def _toolkit_which(x: str) -> str | None:
    return "/usr/bin/nvidia-container-toolkit" if "nvidia" in x else None


@mock.patch("shutil.which", side_effect=_toolkit_which)
def test_detect_container_runtime_toolkit_no_docker(mock_which):
    runtime, has_toolkit = detect._detect_container_runtime()
    assert runtime is None
    assert has_toolkit is True


@mock.patch("shutil.which", return_value=None)
def test_detect_container_runtime_none(mock_which):
    runtime, has_toolkit = detect._detect_container_runtime()
    assert runtime is None
    assert has_toolkit is False


# ===========================================================================
# detect() — integration
# ===========================================================================


@mock.patch.object(detect, "_detect_container_runtime")
@mock.patch.object(detect, "_get_cpu_ram_gb")
@mock.patch.object(detect, "_detect_os_and_wsl")
def test_detect_no_gpu(
    mock_os_wsl,
    mock_ram,
    mock_ctr,
):
    """detect() returns a HostCapabilities with no GPUs when none are found."""
    mock_os_wsl.return_value = ("Linux", False)
    mock_ram.return_value = 32.0
    mock_ctr.return_value = ("docker", True)

    host = detect.detect()

    assert host.os == "Linux"
    assert host.is_wsl is False
    assert host.cpu_ram_gb == 32.0
    assert host.container_runtime == "docker"
    assert host.has_nvidia_container_toolkit is True
    assert host.gpus == ()


@mock.patch.object(detect, "_probe_gpu_pynvml")
@mock.patch.object(detect, "_probe_gpu_smi")
@mock.patch.object(detect, "_detect_container_runtime")
@mock.patch.object(detect, "_get_cpu_ram_gb")
@mock.patch.object(detect, "_detect_os_and_wsl")
def test_detect_primary_path(
    mock_os_wsl,
    mock_ram,
    mock_ctr,
    mock_smi,
    mock_pynvml,
):
    """detect() uses pynvml (primary) before falling back to nvidia-smi."""
    mock_os_wsl.return_value = ("Windows", False)
    mock_ram.return_value = 16.0
    mock_ctr.return_value = (None, False)
    # Primary path returns data
    mock_pynvml.return_value = [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "vram_total_mb": 24576,
            "vram_free_mb": 22000,
            "compute_cap_major": 8,
            "compute_cap_minor": 9,
            "driver_version": "555.99",
        }
    ]
    mock_smi.return_value = None  # Should not be called

    host = detect.detect()

    assert host.os == "Windows"
    assert host.cpu_ram_gb == 16.0
    assert len(host.gpus) == 1
    assert host.gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert host.gpus[0].vram_total_mb == 24576
    assert host.gpus[0].arch is contracts.GpuArch.ADA
    assert host.gpus[0].mem_bandwidth_gbs == 1008.0
    # Primary was called, fallback was not
    mock_pynvml.assert_called_once()
    mock_smi.assert_not_called()


@mock.patch.object(detect, "_probe_gpu_pynvml")
@mock.patch.object(detect, "_probe_gpu_smi")
@mock.patch.object(detect, "_detect_container_runtime")
@mock.patch.object(detect, "_get_cpu_ram_gb")
@mock.patch.object(detect, "_detect_os_and_wsl")
def test_detect_fallback_path(
    mock_os_wsl,
    mock_ram,
    mock_ctr,
    mock_smi,
    mock_pynvml,
):
    """detect() falls back to nvidia-smi when pynvml returns nothing."""
    mock_os_wsl.return_value = ("Linux", True)
    mock_ram.return_value = 64.0
    mock_ctr.return_value = ("docker", True)
    # Primary returns nothing
    mock_pynvml.return_value = None
    mock_smi.return_value = [
        {
            "index": 0,
            "name": "NVIDIA A100-SXM4-80GB",
            "vram_total_mb": 81920,
            "vram_free_mb": 76800,
            "compute_cap_major": 8,
            "compute_cap_minor": 0,
            "driver_version": "550.54",
        }
    ]

    host = detect.detect()

    assert len(host.gpus) == 1
    assert host.gpus[0].name == "NVIDIA A100-SXM4-80GB"
    assert host.gpus[0].arch is contracts.GpuArch.AMPERE
    assert host.gpus[0].mem_bandwidth_gbs == 1555.0
    mock_pynvml.assert_called_once()
    mock_smi.assert_called_once()


@mock.patch.object(detect, "_probe_gpu_pynvml")
@mock.patch.object(detect, "_probe_gpu_smi")
@mock.patch.object(detect, "_detect_container_runtime")
@mock.patch.object(detect, "_get_cpu_ram_gb")
@mock.patch.object(detect, "_detect_os_and_wsl")
def test_detect_wsl_with_rtx_3050_workflow(
    mock_os_wsl,
    mock_ram,
    mock_ctr,
    mock_smi,
    mock_pynvml,
):
    """Simulate the reference profile — RTX 3050 6GB Laptop under WSL.

    This mirrors the worked example from HER-8.
    """
    mock_os_wsl.return_value = ("Linux", True)
    mock_ram.return_value = 16.0
    mock_ctr.return_value = ("docker", True)
    mock_pynvml.return_value = [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
            "vram_total_mb": 6144,
            "vram_free_mb": 5120,
            "compute_cap_major": 8,
            "compute_cap_minor": 6,
            "driver_version": "555.99",
        }
    ]

    host = detect.detect()

    gpu = host.gpus[0]
    assert gpu.arch is contracts.GpuArch.AMPERE
    assert gpu.compute_capability == (8, 6)
    assert gpu.vram_total_mb == 6144
    assert gpu.vram_free_mb == 5120
    assert gpu.mem_bandwidth_gbs == 170.0
    assert gpu.supports_fp8 is False  # Ampere
    assert gpu.supports_int8 is True  # CC >= 7.5
    assert gpu.driver_version == "555.99"
    # Full host picture
    assert host.os == "Linux"
    assert host.is_wsl is True
    assert host.container_runtime == "docker"
    assert host.has_nvidia_container_toolkit is True


# ===========================================================================
# CPU RAM detection
# ===========================================================================


@mock.patch("sys.platform", "linux")
def test_get_cpu_ram_linux():
    meminfo = "MemTotal:       32829248 kB\nMemFree:        12000000 kB\n"
    with mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
        ram = detect._get_cpu_ram_gb()
    assert ram == pytest.approx(31.3, rel=0.1)


@mock.patch("sys.platform", "win32")
@mock.patch("subprocess.run")
def test_get_cpu_ram_windows_wmic(mock_run):
    """On Windows, prefer wmic output."""
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Capacity\n17179869184\n17179869184\n"
    ram = detect._get_cpu_ram_gb()
    assert ram == pytest.approx(32.0, rel=0.1)


@mock.patch("sys.platform", "win32")
def test_get_cpu_ram_windows_powershell_fallback():
    """If wmic fails, try PowerShell as fallback."""
    calls = 0

    def _run(args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FileNotFoundError("wmic not found")
        result = mock.MagicMock()
        result.returncode = 0
        result.stdout = "34359738368\n"
        return result

    with mock.patch("subprocess.run", side_effect=_run):
        ram = detect._get_cpu_ram_gb()
    assert ram == pytest.approx(32.0, rel=0.1)
    assert calls == 2


# ===========================================================================
# Edge cases
# ===========================================================================


def test_lookup_bandwidth_no_match():
    """Unknown GPUs produce None bandwidths."""
    mem, pcie = detect._lookup_bandwidth("Intel Arc A770")
    assert mem is None
    assert pcie is None


def test_lookup_bandwidth_longest_key_wins():
    """'RTX 3080 Ti' must match the Ti entry, not the non-Ti entry."""
    mem3080, _ = detect._lookup_bandwidth("NVIDIA GeForce RTX 3080")
    mem3080ti, _ = detect._lookup_bandwidth("NVIDIA GeForce RTX 3080 Ti")
    assert mem3080 != mem3080ti
    assert mem3080ti == 912.0
