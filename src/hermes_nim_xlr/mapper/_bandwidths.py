"""GPU bandwidth and PCIe lookup tables.

Memory bandwidth (GB/s) and typical PCIe generation / lane count for common
NVIDIA GPUs.  Keyed by substrings matched against the GPU name string.
Unknown GPUs produce None for bandwidths.
"""

from __future__ import annotations

import re

# PCIe per-lane bandwidth in GB/s (bidirectional, 1e9 bytes).
_PCIE_GEN_BW: dict[int, float] = {
    1: 0.250,
    2: 0.500,
    3: 0.985,
    4: 1.969,
    5: 3.938,
    6: 7.563,
}


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
    matched_entry: tuple[float | None, int | None, int | None] | None = None

    for key, entry in _BANDWIDTH_LOOKUP.items():
        key_tokens = key.lower().split()
        if not key_tokens:
            continue
        # Every word in the key must appear somewhere in the GPU name.
        if all(t in name_tokens for t in key_tokens):
            if len(key) > len(matched_key):
                matched_key = key
                matched_entry = entry

    if matched_entry is None:
        return None, None

    mem_bw, pcie_gen, pcie_lanes = matched_entry
    if pcie_gen is not None and pcie_lanes is not None:
        pcie_bw = _PCIE_GEN_BW.get(pcie_gen) * pcie_lanes
    else:
        pcie_bw = None
    return mem_bw, pcie_bw


# ---------------------------------------------------------------------------#
# Static bandwidth lookup table
# ---------------------------------------------------------------------------#

_BANDWIDTH_ENTRY = tuple[float | None, int | None, int | None]
"""A (mem_bandwidth_gbs, pcie_gen, pcie_lanes) row."""

_BANDWIDTH_LOOKUP: dict[str, _BANDWIDTH_ENTRY] = {
    # ---- GeForce RTX 30-series (Ampere) ----
    "RTX 3050": (224.0, 4, 8),
    "RTX 3050 Laptop": (170.0, 4, 8),
    "RTX 3060": (360.0, 4, 16),
    "RTX 3060 Laptop": (192.0, 4, 8),
    "RTX 3060 Ti": (448.0, 4, 16),
    "RTX 3070": (448.0, 4, 16),
    "RTX 3070 Laptop": (384.0, 4, 8),
    "RTX 3070 Ti": (448.0, 4, 16),
    "RTX 3080": (760.0, 4, 16),
    "RTX 3080 Laptop": (384.0, 4, 8),
    "RTX 3080 Ti": (912.0, 4, 16),
    "RTX 3090": (936.0, 4, 16),
    "RTX 3090 Ti": (1008.0, 4, 16),
    # ---- GeForce RTX 40-series (Ada Lovelace) ----
    "RTX 4050 Laptop": (192.0, 4, 8),
    "RTX 4060": (272.0, 4, 8),
    "RTX 4060 Laptop": (256.0, 4, 8),
    "RTX 4060 Ti": (288.0, 4, 8),
    "RTX 4070": (504.0, 4, 16),
    "RTX 4070 Laptop": (256.0, 4, 8),
    "RTX 4070 Ti": (576.0, 4, 16),
    "RTX 4070 Ti Super": (672.0, 4, 16),
    "RTX 4080": (736.0, 4, 16),
    "RTX 4080 Laptop": (512.0, 4, 8),
    "RTX 4080 Super": (736.0, 4, 16),
    "RTX 4090": (1008.0, 4, 16),
    "RTX 4090 Laptop": (576.0, 4, 8),
    # ---- GeForce RTX 50-series (Blackwell) ----
    "RTX 5060": (512.0, 5, 8),
    "RTX 5070": (672.0, 5, 16),
    "RTX 5080": (960.0, 5, 16),
    "RTX 5090": (1792.0, 5, 16),
    "RTX 5090 Laptop": (880.0, 5, 8),
    # ---- Enterprise / Pro ----
    "A100": (1555.0, 4, 16),
    "H100": (3352.0, 5, 16),
    "B200": (4304.0, 5, 16),
    "T4": (320.0, 3, 16),
    "L4": (300.0, 4, 16),
    "L40S": (864.0, 4, 16),
    "A10": (600.0, 4, 16),
    "A30": (933.0, 4, 16),
    "A40": (696.0, 4, 16),
}
