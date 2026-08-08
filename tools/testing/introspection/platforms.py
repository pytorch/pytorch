"""Platform descriptors.

A Platform is a declarative description of a CI target. Applying it (see
collector.apply_descriptor) monkeypatches the small, centralized device-capability
surface -- `torch.cuda`/`torch.xpu` probes, the `common_cuda` SM / PLATFORM_SUPPORTS_*
flags, the device `*TestBase.setUpClass`, and `device_type_test_bases` -- so that test
generation and skip evaluation run as pure Python for the declared platform, with no
real-driver calls. This lets one host answer for many platforms.

The descriptor surface was validated to be bounded and centralized: see the spike
results in the design doc. A missing entry fails loudly (a real driver call raising),
so descriptors are discoverable and maintainable, never silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Platform:
    name: str
    # Primary accelerator device for this platform. CPU device-generic tests are also
    # generated/run on accelerator platforms (a CUDA box has a CPU too), matching CI.
    device_type: str  # "cpu" | "cuda" | "mps" | "xpu"  (rocm uses device_type "cuda")
    rocm: bool = False
    # CUDA/ROCm compute capability the descriptor declares (drives SMxxOrLater gates).
    cuda_capability: tuple[int, int] | None = None
    cudnn_version: int = 90100
    # common_cuda PLATFORM_SUPPORTS_* overrides (name -> bool). Defaults are filled by
    # the factories below from the declared capability.
    caps: dict[str, bool] = field(default_factory=dict)
    # Extra subprocess env (e.g. PYTORCH_TEST_WITH_ROCM=1).
    env: dict[str, str] = field(default_factory=dict)

    def subprocess_env(self) -> dict[str, str]:
        # Hide any real accelerator so an un-stubbed probe fails loudly instead of
        # silently using the host device (keeps simulation deterministic + descriptors
        # honest). The in-process descriptor supplies the simulated values.
        e = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
        e.update(self.env)
        return e


# Overrides for the PLATFORM_SUPPORTS_* flags that the descriptor cannot let
# common_cuda compute for itself. Everything else is deliberately absent: those
# predicates read only torch.cuda.get_device_capability and the SM* LazyVals, all
# of which collector.apply_descriptor patches, so they self-heal to the right
# value for the simulated capability. Re-deriving them here would be a second
# copy of common_cuda's truth table, free to drift without anything noticing.
def _cuda_caps(capability: tuple[int, int], rocm: bool) -> dict[str, bool]:
    return {
        # A plain bool, not a LazyVal: frozen at common_cuda import time, when the
        # descriptor has hidden the GPU, so it cannot self-heal.
        "PLATFORM_SUPPORTS_FUSED_SDPA": not rocm,
        "PLATFORM_SUPPORTS_CK_SDPA": rocm,
        # Read the build, not the device: torch.__config__.show() / cusparselt.
        "PLATFORM_SUPPORTS_MXFP8_GROUPED_GEMM": capability >= (10, 0) and not rocm,
        "PLATFORM_SUPPORTS_FP8_SPARSE": False,
        # Probe the driver via green_contexts._ensure_supported().
        "PLATFORM_SUPPORTS_GREEN_CONTEXT": False,
        "PLATFORM_SUPPORTS_WORKQUEUE_CONFIG": False,
    }


def cpu(name: str = "linux-cpu") -> Platform:
    return Platform(name=name, device_type="cpu")


def cuda(name: str, capability: tuple[int, int] = (8, 0), **caps: bool) -> Platform:
    d = _cuda_caps(capability, rocm=False)
    d.update(caps)
    return Platform(name=name, device_type="cuda", cuda_capability=capability, caps=d)


def rocm(name: str = "linux-rocm", capability: tuple[int, int] = (9, 4)) -> Platform:
    d = _cuda_caps(capability, rocm=True)
    return Platform(
        name=name,
        device_type="cuda",
        rocm=True,
        cuda_capability=capability,
        caps=d,
        env={"PYTORCH_TEST_WITH_ROCM": "1"},
    )


def mps(name: str = "macos-mps") -> Platform:
    return Platform(name=name, device_type="mps")


def xpu(name: str = "linux-xpu") -> Platform:
    return Platform(name=name, device_type="xpu")


# Built-in registry. Extend / generate from .github workflows later.
REGISTRY: dict[str, Platform] = {
    p.name: p
    for p in [
        cpu("linux-cpu"),
        cuda("linux-cuda-sm80", (8, 0)),
        cuda("linux-cuda-sm86", (8, 6)),
        cuda("linux-cuda-sm90", (9, 0)),
        cuda("linux-cuda-sm100", (10, 0)),
        rocm("linux-rocm"),
        mps("macos-mps"),
        xpu("linux-xpu"),
    ]
}


def get(name: str) -> Platform:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown platform {name!r}; known: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]


@dataclass(frozen=True)
class Config:
    """The CI TEST_CONFIG dimension: which files are selected, and the env flags
    (PYTORCH_TEST_WITH_*) that change generation/skip. `options` are run_test
    arg-namespace overrides consumed by run_test.get_selected_tests."""

    name: str
    options: dict[str, object] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


CONFIGS: dict[str, Config] = {
    c.name: c
    for c in [
        Config("default"),
        Config("distributed", options={"distributed_tests": True}),
        Config("functorch", options={"functorch": True}),
        Config("inductor", options={"include_inductor_core_tests": True}),
        Config(
            "dynamo",
            options={"include_dynamo_core_tests": True},
            env={"PYTORCH_TEST_WITH_DYNAMO": "1"},
        ),
        Config("slow", env={"PYTORCH_TEST_WITH_SLOW": "1"}),
        Config("crossref", env={"PYTORCH_TEST_WITH_CROSSREF": "1"}),
        Config("mps", options={"mps": True}),
    ]
}


def get_config(name: str) -> Config:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; known: {', '.join(sorted(CONFIGS))}")
    return CONFIGS[name]


@dataclass(frozen=True)
class Job:
    """A CI target = (platform device descriptor, config). Names as 'platform/config'."""

    platform: Platform
    config: Config

    @property
    def name(self) -> str:
        return f"{self.platform.name}/{self.config.name}"

    def subprocess_env(self) -> dict[str, str]:
        e = self.platform.subprocess_env()
        e.update(self.config.env)
        return e


def get_job(name: str) -> Job:
    """Parse 'platform/config' (config defaults to 'default')."""
    if "/" in name:
        pname, cname = name.rsplit("/", 1)
    else:
        pname, cname = name, "default"
    return Job(get(pname), get_config(cname))
