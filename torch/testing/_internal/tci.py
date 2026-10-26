import re
from dataclasses import dataclass
from enum import Enum, EnumMeta


class Unsupported(Exception):
    """Raised by ``current()`` when the running host is not a known member."""


class _ConstraintMeta(EnumMeta):
    """Inject each member's attribute name as its leading value.

    Members declare only their payload (``()`` when they have none); this
    metaclass prepends the member name so the marker string is always derived
    from the name and never has to be repeated by hand. Injecting the unique
    name also keeps members that share a payload (e.g. two gpus with the same
    arch and sdks) from collapsing into enum aliases.
    """

    def __new__(mcs, cls_name, bases, ns, **kwds):
        # ns._member_names is the enum machinery's own record of which keys are
        # members; rewrite their values before EnumMeta builds the members.
        for member in list(ns._member_names):
            payload = ns[member]
            if not isinstance(payload, tuple):
                payload = (payload,)
            dict.__setitem__(ns, member, (member, *payload))
        return super().__new__(mcs, cls_name, bases, ns, **kwds)


class _Constraint(Enum, metaclass=_ConstraintMeta):
    def __new__(cls, raw_value: str):
        obj = object.__new__(cls)
        obj._value_ = cls._make_value(raw_value)
        return obj

    @classmethod
    def _make_value(cls, raw_value: str) -> str:
        label = "_".join(
            match.group(0).lower()
            for match in re.finditer(
                r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", cls.__name__
            )
        )
        return f"tci.{label}:{raw_value}"

    def __str__(self) -> str:
        return self.value


class _Detectable:
    """Mixin for constraints that can be detected on the running host.

    Subclasses implement ``current()`` to return the member matching the
    running host. ``is_current()`` derives from it and is ``False`` whenever
    the host does not match this member.
    """

    @classmethod
    def current(cls) -> "_Constraint | None":
        raise NotImplementedError

    def is_current(self) -> bool:
        try:
            return self is type(self).current()
        except Unsupported:
            return False


class schedule(_Constraint):
    pull = ()
    periodic = ()


class size(_Constraint):
    small = ()
    medium = ()
    large = ()


class os(_Detectable, _Constraint):
    linux = ()
    windows = ()
    macos = ()

    @classmethod
    def current(cls) -> "os":
        import sys

        if sys.platform.startswith("linux"):
            return cls.linux
        if sys.platform in ("win32", "cygwin"):
            return cls.windows
        if sys.platform == "darwin":
            return cls.macos
        raise Unsupported(f"unsupported os: {sys.platform}")


class cpu(_Detectable, _Constraint):
    x86_64 = ()
    arm64 = ()
    s390x = ()

    @classmethod
    def current(cls) -> "cpu":
        import platform

        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            return cls.x86_64
        if machine in ("aarch64", "arm64"):
            return cls.arm64
        if machine == "s390x":
            return cls.s390x
        raise Unsupported(f"unsupported cpu: {machine}")


class gpu(_Detectable, _Constraint):
    arch: str
    sdks: tuple[str, ...]

    def __new__(cls, raw_value: str, arch: str, sdks: tuple[str, ...]):
        obj = object.__new__(cls)
        obj._value_ = cls._make_value(raw_value)
        obj.arch = arch
        obj.sdks = sdks
        return obj

    # cuda
    cuda_t4 = "sm75", ("12.6", "12.8", "13.0", "13.2", "13.4")
    cuda_a100 = "sm80", ("12.6", "12.8", "13.0", "13.2")
    cuda_a10g = "sm86", ("12.6", "12.8", "13.0", "13.2", "13.4")
    cuda_l4 = "sm89", ("12.6", "12.8", "13.0", "13.2", "13.4")
    cuda_h100 = "sm90", ("12.6", "12.8", "13.0", "13.2")
    cuda_h200 = "sm90", ("12.6", "12.8", "13.0", "13.2")
    cuda_b200 = "sm100", ("12.8", "13.0", "13.2", "13.4")
    cuda_gb200 = "sm100", ("12.8", "13.0", "13.2")
    cuda_gb300 = "sm103", ("13.2", "13.4")
    # rocm
    rocm_mi200 = "gfx90a", ("7.1", "7.2")
    rocm_mi210 = "gfx90a", ("7.1", "7.2", "10.1", "10.2")
    rocm_mi250 = "gfx90a", ("7.1", "7.2")
    rocm_mi300 = "gfx942", ("7.1", "7.2", "10.1", "10.2")
    rocm_mi350 = "gfx950", ("7.1", "7.2", "10.1", "10.2")
    rocm_navi31 = "gfx1100", ("7.1", "7.2", "10.1")
    # xpu
    xpu_pvc = "xe12", ("2026.0",)
    xpu_bmg = "xe20", ("2026.0",)
    # MPS is versioned with macOS, not a separate SDK.
    mps_m1 = "apple7", ()
    mps_m2 = "apple8", ()

    @classmethod
    def current(cls) -> "gpu | None":
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            if torch.version.hip:
                vendor = "rocm"
                # gcnArchName can be "gfx90a:sramecc+:xnack-"; keep the base.
                arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            else:
                vendor = "cuda"
                major, minor = torch.cuda.get_device_capability()
                arch = f"sm{major}{minor}"
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            # xpu exposes no stable arch string; match on the device name alone.
            vendor, arch, name = "xpu", None, torch.xpu.get_device_name()
        elif torch.backends.mps.is_available():
            vendor, arch, name = "mps", None, torch.backends.mps.get_name()
        else:
            # not all machines have a gpu; absence is not an error
            return None
        # Primary match is arch -- the capability the enum records. When several
        # members share an arch (e.g. sm90 -> cuda_h100/cuda_h200) disambiguate
        # by the product token in the device name, longest match winning.
        candidates = [
            member
            for member in cls
            if member.name.startswith(f"{vendor}_")
            and (arch is None or member.arch == arch)
        ]
        if len(candidates) > 1:
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            candidates = [
                member
                for member in candidates
                if member.name.split("_", 1)[1] in normalized
            ]
        if not candidates:
            raise Unsupported(f"unsupported gpu: {name}")
        return max(candidates, key=lambda member: len(member.name))


class platform(_Detectable, _Constraint):
    os: os
    cpu: cpu
    gpu: gpu | None

    def __new__(cls, raw_value: str, os: os, cpu: cpu, gpu: gpu | None = None):
        obj = object.__new__(cls)
        obj._value_ = cls._make_value(raw_value)
        obj.os = os
        obj.cpu = cpu
        obj.gpu = gpu
        return obj

    # OS/CPU/GPU combinations observed in ci.environments.
    linux_x86_64 = os.linux, cpu.x86_64
    linux_arm64 = os.linux, cpu.arm64
    linux_x86_64_cuda_t4 = os.linux, cpu.x86_64, gpu.cuda_t4
    linux_x86_64_cuda_a10g = os.linux, cpu.x86_64, gpu.cuda_a10g
    linux_x86_64_cuda_l4 = os.linux, cpu.x86_64, gpu.cuda_l4
    linux_x86_64_cuda_h100 = os.linux, cpu.x86_64, gpu.cuda_h100
    linux_x86_64_cuda_b200 = os.linux, cpu.x86_64, gpu.cuda_b200
    linux_arm64_cuda_gb300 = os.linux, cpu.arm64, gpu.cuda_gb300
    linux_x86_64_rocm_mi210 = os.linux, cpu.x86_64, gpu.rocm_mi210
    linux_x86_64_rocm_mi300 = os.linux, cpu.x86_64, gpu.rocm_mi300
    linux_x86_64_rocm_mi350 = os.linux, cpu.x86_64, gpu.rocm_mi350
    linux_x86_64_rocm_navi31 = os.linux, cpu.x86_64, gpu.rocm_navi31
    macos_arm64_mps_m1 = os.macos, cpu.arm64, gpu.mps_m1
    macos_arm64_mps_m2 = os.macos, cpu.arm64, gpu.mps_m2
    windows_x86_64 = os.windows, cpu.x86_64

    @classmethod
    def current(cls) -> "platform":
        current_os = os.current()
        current_cpu = cpu.current()
        # gpu is optional: a host with no (or an unrecognized) accelerator can
        # still match a member whose gpu is None.
        try:
            current_gpu = gpu.current()
        except Unsupported:
            current_gpu = None
        for member in cls:
            if (
                member.os is current_os
                and member.cpu is current_cpu
                and member.gpu is current_gpu
            ):
                return member
        raise Unsupported(
            f"unsupported platform: {current_os}, {current_cpu}, {current_gpu}"
        )


@dataclass(frozen=True)
class _test:
    platforms: tuple[platform, ...] | None
    add: tuple[platform, ...]
    exclude: tuple[platform, ...]
    devices: tuple[str, ...] | None
    exclude_devices: tuple[str, ...]
    size: size | None
    schedule: schedule | None

    def __post_init__(self):
        for members in (self.platforms, self.add, self.exclude):
            if members is not None and not all(
                isinstance(member, platform) for member in members
            ):
                raise TypeError(f"expected tci.platform members, got {members!r}")
        for members in (self.devices, self.exclude_devices):
            if members is not None and not all(
                isinstance(member, str)
                and re.fullmatch(r"[a-z][a-z0-9_]*", member)
                and member != "none"
                for member in members
            ):
                raise TypeError(f"expected tci.device names, got {members!r}")
        if self.size is not None and not isinstance(self.size, size):
            raise TypeError(f"expected tci.size, got {self.size!r}")
        if self.schedule is not None and not isinstance(self.schedule, schedule):
            raise TypeError(f"expected tci.schedule, got {self.schedule!r}")

    def __call__(self, target):
        # Read only own declarations so subclasses do not copy inherited policy.
        target._tci = (*vars(target).get("_tci", ()), self)
        return target


def test(
    *,
    platform: platform | tuple[platform, ...] | None = None,
    add: platform | tuple[platform, ...] = (),
    exclude: platform | tuple[platform, ...] = (),
    device: str | tuple[str, ...] | None = None,
    exclude_device: str | tuple[str, ...] = (),
    size: size | None = None,
    schedule: schedule | None = None,
) -> _test:
    """Declare test placement without changing the decorated callable.

    ``platform`` replaces inherited platforms, then ``add`` and ``exclude``
    modify that selection. Unspecified fields inherit from the test class.
    An empty ``platform=()`` explicitly selects no platforms.
    ``device`` selects generated device variants and replaces inherited device
    restrictions; ``exclude_device`` removes variants from that selection.
    """
    platforms = (
        platform if platform is None or isinstance(platform, tuple) else (platform,)
    )
    additions = add if isinstance(add, tuple) else (add,)
    exclusions = exclude if isinstance(exclude, tuple) else (exclude,)
    devices = device if device is None or isinstance(device, tuple) else (device,)
    excluded_devices = (
        exclude_device if isinstance(exclude_device, tuple) else (exclude_device,)
    )
    return _test(
        platforms, additions, exclusions, devices, excluded_devices, size, schedule
    )


@dataclass(frozen=True)
class _ResolvedTest:
    platforms: tuple[platform, ...]
    size: size
    schedule: schedule
    devices: tuple[str, ...] | None
    excluded_devices: tuple[str, ...]

    def supports_device(self, device: str, *, bypass: bool = False) -> bool:
        return (
            bypass or self.devices is None or device in self.devices
        ) and device not in self.excluded_devices

    @property
    def markers(self) -> tuple[str, ...]:
        markers = tuple(
            str(member) for member in (*self.platforms, self.size, self.schedule)
        )
        if self.devices is not None:
            markers += tuple(
                f"tci.device:{device}" for device in self.devices or ("none",)
            )
        return markers + tuple(
            f"tci.exclude_device:{device}" for device in self.excluded_devices
        )


def resolve(target, cls: type | None = None) -> _ResolvedTest | None:
    """Resolve class defaults and method overrides into effective test metadata.

    Classes follow Python's method resolution order, with the nearest class
    taking precedence. Stacked decorators apply from the innermost outward.
    An unmarked function or class returns ``None``.
    """
    if isinstance(target, type):
        cls, target = target, None
    elif cls is None and (owner := getattr(target, "__self__", None)) is not None:
        cls = owner if isinstance(owner, type) else type(owner)
    declarations = []
    if cls is not None:
        for base in reversed(cls.__mro__):
            declarations.extend(vars(base).get("_tci", ()))
    if target is not None:
        declarations.extend(getattr(target, "__dict__", {}).get("_tci", ()))
    if not declarations:
        return None

    platforms: tuple[platform, ...] = ()
    devices: tuple[str, ...] | None = None
    excluded_devices: tuple[str, ...] = ()
    resolved_size, resolved_schedule = size.small, schedule.pull
    for declaration in declarations:
        if declaration.platforms is not None:
            platforms = declaration.platforms
        platforms = tuple(
            member
            for member in dict.fromkeys((*platforms, *declaration.add))
            if member not in declaration.exclude
        )
        if declaration.devices is not None:
            devices = tuple(dict.fromkeys(declaration.devices))
            excluded_devices = ()
        excluded_devices = tuple(
            dict.fromkeys((*excluded_devices, *declaration.exclude_devices))
        )
        if declaration.size is not None:
            resolved_size = declaration.size
        if declaration.schedule is not None:
            resolved_schedule = declaration.schedule
    return _ResolvedTest(
        platforms, resolved_size, resolved_schedule, devices, excluded_devices
    )


def current_platform() -> platform | None:
    """Select the rollout platform without initializing an accelerator runtime.

    An explicit scheduler platform takes precedence. Automatic selection covers
    Linux x86_64 CPU and L4 hosts; unknown and unmigrated hosts return ``None``.
    """
    import os as os_module

    explicit = os_module.environ.get("PYTORCH_TEST_TCI_PLATFORM")
    if explicit is not None:
        try:
            return platform[explicit]
        except KeyError as error:
            raise ValueError(
                f"invalid PYTORCH_TEST_TCI_PLATFORM: {explicit!r}"
            ) from error

    try:
        if os.current() is not os.linux or cpu.current() is not cpu.x86_64:
            return None
    except Unsupported:
        return None

    import torch

    if torch.version.hip or getattr(torch.version, "xpu", None):
        return None
    if torch.cuda._is_compiled():
        try:
            count = torch.cuda._device_count_nvml()
            if count < 0:
                return None
            if count:
                from torch.testing._internal.torchci.environment import _nvml_device

                name = re.sub(r"[^a-z0-9]", "", _nvml_device()[0].lower())
                if name in ("l4", "nvidial4", "teslal4"):
                    return platform.linux_x86_64_cuda_l4
                return None
        except Exception:
            return None
    return platform.linux_x86_64


def is_eligible(
    metadata: _ResolvedTest | None,
    *,
    platform: platform | None = None,
    device: str | None = None,
    bypass: bool = False,
    strict: bool = False,
) -> bool:
    """Apply device constraints and the active platform's selection policy.

    Normal runners preserve tests on unmigrated or unknown hosts. Schedulers
    use ``strict=True`` to require an explicit declaration of their platform.
    ``device`` is the generated variant, not the host's accelerator.
    """
    if metadata is None:
        return not strict
    if isinstance(device, str) and not metadata.supports_device(device, bypass=bypass):
        return False
    if strict or platform in _ROLLOUT_PLATFORMS:
        return platform in metadata.platforms
    return True


_ROLLOUT_PLATFORMS = (platform.linux_x86_64, platform.linux_x86_64_cuda_l4)
