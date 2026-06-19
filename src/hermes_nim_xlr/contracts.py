"""The ExecutionPlan contract — frozen dataclasses + enums every downstream layer
consumes.

Pure stdlib, zero hardware/heavy-import dependencies (importable on any host,
no CUDA/torch). Field names for KvCacheConfig map 1:1 onto TensorRT-LLM's real
KVCacheConfig (enable_block_reuse, free_gpu_memory_fraction,
host_cache_size_bytes) per spec.md §1.2 — no invented flags.
"""

from dataclasses import dataclass
from enum import Enum


class GpuArch(Enum):
    AMPERE = "ampere"
    ADA = "ada"
    HOPPER = "hopper"
    BLACKWELL = "blackwell"
    OTHER = "other"


class WeightQuant(Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4_AWQ = "int4_awq"


class KvDtype(Enum):
    FP16 = "fp16"
    INT8 = "int8"
    FP8 = "fp8"


class SpecDecode(Enum):
    NONE = "none"
    NGRAM = "ngram"
    EAGLE = "eagle"
    DRAFT_TARGET = "draft_target"


class BackendKind(Enum):
    TRTLLM = "tensorrt_llm"
    LLAMACPP = "llama_cpp"
    MLX = "mlx"


class BringUp(Enum):
    NATIVE_LINUX = "native_linux"
    WSL2_DOCKER = "wsl2_docker"
    NATIVE_WINDOWS = "native_windows"


class Objective(Enum):
    THROUGHPUT_FIRST = "throughput_first"
    QUALITY_FIRST = "quality_first"


@dataclass(frozen=True)
class GpuCapabilities:
    index: int
    name: str
    arch: GpuArch
    compute_capability: tuple[int, int]
    vram_total_mb: int
    vram_free_mb: int
    mem_bandwidth_gbs: float | None
    pcie_bandwidth_gbs: float | None
    supports_fp8: bool
    supports_int8: bool
    supports_cuda_graphs: bool
    driver_version: str


@dataclass(frozen=True)
class HostCapabilities:
    os: str
    is_wsl: bool
    cpu_ram_gb: float
    container_runtime: str | None
    has_nvidia_container_toolkit: bool
    gpus: tuple[GpuCapabilities, ...]


@dataclass(frozen=True)
class ModelChoice:
    repo: str
    params_b: float
    weight_quant: WeightQuant
    est_weight_mb: int
    n_layers: int


@dataclass(frozen=True)
class KvCacheConfig:  # <-> TensorRT-LLM KVCacheConfig (real field names, spec.md §1.2)
    dtype: KvDtype
    enable_block_reuse: bool
    free_gpu_memory_fraction: float
    host_cache_size_bytes: int


@dataclass(frozen=True)
class DecodeLevers:
    cuda_graphs: bool
    spec_decode: SpecDecode
    draft_model: str | None


@dataclass(frozen=True)
class BackendChoice:
    kind: BackendKind
    bring_up: BringUp
    serve_endpoint: str


@dataclass(frozen=True)
class LayerPlacement:
    total_layers: int
    gpu_layers: int
    cpu_offload_layers: int
    tensor_parallel: int
    pipeline_parallel: int
    note: str


@dataclass(frozen=True)
class ExecutionPlan:
    objective: Objective
    model: ModelChoice
    placement: LayerPlacement
    kv: KvCacheConfig
    levers: DecodeLevers
    backend: BackendChoice
    target_ctx_tokens: int
    est_vram_mb: int
    est_decode_tok_s: tuple[int, int]
    rationale: tuple[str, ...]
    warnings: tuple[str, ...]
