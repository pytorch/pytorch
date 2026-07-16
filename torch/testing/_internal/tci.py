import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum, EnumMeta

import pytest


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
    cuda_t4 = "sm75", ("12.6", "12.8", "13.0", "13.2")
    cuda_a100 = "sm80", ("12.6", "12.8", "13.0", "13.2")
    cuda_a10g = "sm86", ("12.6", "12.8", "13.0", "13.2")
    cuda_l4 = "sm89", ("12.6", "12.8", "13.0", "13.2")
    cuda_h100 = "sm90", ("12.6", "12.8", "13.0", "13.2")
    cuda_h200 = "sm90", ("12.6", "12.8", "13.0", "13.2")
    cuda_b200 = "sm100", ("12.8", "13.0", "13.2")
    cuda_gb200 = "sm100", ("12.8", "13.0", "13.2")
    # rocm
    rocm_mi200 = "gfx90a", ("7.1", "7.2")
    rocm_mi210 = "gfx90a", ("7.1", "7.2")
    rocm_mi250 = "gfx90a", ("7.1", "7.2")
    rocm_mi300 = "gfx942", ("7.1", "7.2")
    rocm_mi350 = "gfx950", ("7.1", "7.2")
    rocm_navi31 = "gfx1100", ("7.1", "7.2")
    # xpu
    xpu_pvc = "xe12", ("2026.0",)
    xpu_bmg = "xe20", ("2026.0",)

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

    linux_x86_cuda_a100 = os.linux, cpu.x86_64, gpu.cuda_a100
    linux_x86_cuda_h100 = os.linux, cpu.x86_64, gpu.cuda_h100
    linux_x86_rocm_mi300 = os.linux, cpu.x86_64, gpu.rocm_mi300
    linux_x86_xpu_pvc = os.linux, cpu.x86_64, gpu.xpu_pvc

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

    @property
    def vendor(self) -> str:
        # cuda_* / rocm_* / xpu_* -> cuda / rocm / xpu ; no gpu -> cpu
        return self.gpu.name.split("_", 1)[0] if self.gpu is not None else "cpu"


@dataclass(frozen=True)
class _test:
    """Collection-time metadata for a test: the platforms it can run on plus
    exactly one schedule and one size. Immutable, so the presets below are
    shared safely and every transform returns a fresh config. Applying the
    instance to a test attaches the constraints as pytest markers.
    """

    platforms: frozenset[platform]
    schedule: schedule
    size: size

    def on(self, platforms: Iterable[platform]) -> "_test":
        return replace(self, platforms=frozenset(platforms))

    def include(self, platforms: Iterable[platform]) -> "_test":
        return replace(self, platforms=self.platforms | frozenset(platforms))

    def exclude(self, platforms: Iterable[platform]) -> "_test":
        return replace(self, platforms=self.platforms - frozenset(platforms))

    def only(self, vendor: str) -> "_test":
        kept = frozenset(p for p in self.platforms if p.vendor == vendor)
        return replace(self, platforms=kept)

    def with_schedule(self, value: schedule) -> "_test":
        return replace(self, schedule=value)

    def with_size(self, value: size) -> "_test":
        return replace(self, size=value)

    def _markers(self) -> list[str]:
        # frozenset is unordered; sort platforms for a deterministic pytestmark
        platforms = sorted(self.platforms, key=lambda p: p.name)
        return [str(p) for p in platforms] + [str(self.schedule), str(self.size)]

    def __call__(self, fn):
        if not self.platforms:
            raise ValueError("tci.test: config selects no platform")
        # Marker names contain '.' and ':' so they are not valid identifiers;
        # getattr on the public factory accepts arbitrary names and handles the
        # function-or-class target for us.
        for marker in self._markers():
            fn = getattr(pytest.mark, marker)(fn)
        return fn


# Sugar derived from the enums themselves so it stays in sync as members change:
# .pull/.periodic/.small/.medium/.large replace the scalar, .only_<vendor> filter.
for _member in (*schedule, *size):
    _setter = "with_schedule" if isinstance(_member, schedule) else "with_size"
    _prop = property(lambda self, m=_member, s=_setter: getattr(self, s)(m))
    setattr(_test, _member.name, _prop)
for _vendor in ("cuda", "rocm", "xpu", "cpu"):
    _prop = property(lambda self, v=_vendor: self.only(v))
    setattr(_test, f"only_{_vendor}", _prop)
del _member, _setter, _vendor, _prop


class test:
    """Namespace of ``tci.test`` presets. Each preset is a starting ``_test``
    config that can be applied directly or refined with the fluent transforms;
    every vendor preset is derived from ``default`` so it never drifts.
    """

    default = _test(frozenset(platform), schedule.pull, size.small)
    cuda = default.only_cuda
    rocm = default.only_rocm
    xpu = default.only_xpu
    cpu = default.only_cpu
