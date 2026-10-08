import json
import subprocess
import sys
from dataclasses import dataclass
from functools import cache


CPU_MAX_CORES = 64  # Modal's per-sandbox limits
MEMORY_MAX_MIB = 344064


@dataclass(frozen=True)
class Flavor:
    name: str
    gpu: str | None  # Modal's gpu argument, for example "H100:2"
    gpu_type: str | None
    gpu_count: int
    cpu: int
    memory_mib: int


# Sandbox GPU allowlist mapped to billing keys; billing also lists non-Sandbox GPUs.
GPU_TYPES = {
    "t4": "t4",
    "l4": "l4",
    "a10": "a10g",
    "l40s": "l40s",
    "a100-40gb": "a100_40gb",
    "a100-80gb": "a100_80gb",
    "h100": "h100",
    "h200": "h200",
    "b200": "b200",
    "b300": "b300",
    "rtx-pro-6000": "rtx6000",
}
# RTX PRO 6000's multi-GPU limit is not documented.
GPU_COUNT_LIMITS = {"a10": 4, "rtx-pro-6000": 1}
FLAVORS = [
    "cpu",
    "cpu-big",
    *[
        f"{gpu}:{count}"
        for gpu in GPU_TYPES
        for count in range(1, GPU_COUNT_LIMITS.get(gpu, 8) + 1)
    ],
]


def parse_flavor(name: str) -> Flavor:
    """One of FLAVORS; GPU flavors get 16 cores and 64 GiB per GPU up to Modal's limits."""
    if name == "cpu":
        return Flavor(name, None, None, 0, 8, 32 * 1024)
    if name == "cpu-big":
        return Flavor(name, None, None, 0, CPU_MAX_CORES, 256 * 1024)
    gpu_type, gpu_count = name.split(":")[0], int(name.split(":")[1])
    cpu = min(16 * gpu_count, CPU_MAX_CORES)
    memory_mib = min(64 * 1024 * gpu_count, MEMORY_MAX_MIB)
    return Flavor(
        name, f"{gpu_type.upper()}:{gpu_count}", gpu_type, gpu_count, cpu, memory_mib
    )


@cache
def billing_rates() -> dict[str, float]:
    """Current Modal list prices, straight from the CLI so they never go stale."""
    result = subprocess.run(
        [sys.executable, "-m", "modal", "billing", "rates", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {key: float(value) for key, value in json.loads(result.stdout).items()}


def hourly_rate(flavor: Flavor) -> float:
    rates = billing_rates()
    gpu_cost = (
        flavor.gpu_count * rates[f"gpu_hour_cost_{GPU_TYPES[flavor.gpu_type]}"]
        if flavor.gpu_type
        else 0.0
    )
    cpu_cost = flavor.cpu * rates["cpu_hour_cost_sandbox"]
    memory_cost = flavor.memory_mib / 1024 * rates["mem_gib_hour_cost_sandbox"]
    return round(gpu_cost + cpu_cost + memory_cost, 2)
