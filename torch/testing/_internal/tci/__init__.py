"""tci: the constraints a test puts on the machines it can run on. A test with none
runs everywhere, and each one takes machines away:

    @tci.only(tci.cuda, tci.rocm)  # only with NVIDIA or AMD GPUs
    @tci.exclude(tci.rocm, reason="...")  # not with AMD GPUs
    @tci.needs(tci.gpus >= 2, on=tci.gpu)  # if it has a GPU, two or more
    def test_all_gather(self): ...

A decorator only stores its constraint on the test or the test class under it: it
never skips or wraps. Every level's constraints apply: a test's own, its class's
and those of the class's bases. Whatever runs the tests judges them against its
machine, as pytest does with plugin.py. This module imports no torch, so anything
can import it."""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass
from functools import reduce
from typing import Any, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

# Each accelerator's devices: tci's name for one, and what the name its driver
# reports starts with, so an "NVIDIA H100 80GB HBM3" is a cuda.h100
DEVICES = {
    "cuda": {
        "t4": "Tesla T4",
        "a10": "NVIDIA A10",
        "a10g": "NVIDIA A10G",
        "a100": "NVIDIA A100",
        "l4": "NVIDIA L4",
        "l40s": "NVIDIA L40S",
        "h100": "NVIDIA H100",
        "h200": "NVIDIA H200",
        "b200": "NVIDIA B200",
        "b300": "NVIDIA B300",
        "gb300": "NVIDIA GB300",
        "rtx_pro_6000": "NVIDIA RTX PRO 6000",
    },
    "rocm": {
        "mi210": "AMD Instinct MI210",
        "mi300x": "AMD Instinct MI300X",
        "mi350x": "AMD Instinct MI350X",
        "rx7900": "AMD Radeon RX 7900",
    },
    "xpu": {},
    "mps": {"m1": "Apple M1", "m2_pro": "Apple M2 Pro"},
}
_ATTR = "_tci"  # on a test or a test class: the constraints written right on it


@dataclass(frozen=True)
class Machine:
    """What a constraint is judged against. Machine() has no GPU."""

    accelerator: str = ""  # what its GPUs run on: cuda, rocm, xpu or mps
    gpus: int = 0
    device: str = ""  # the first GPU's name, as its driver reports it
    gpu_memory: float | None = None  # GiB on each GPU; None if nothing reports it
    capability: tuple[int, int] = (0, 0)  # CUDA compute capability: 9.0 is (9, 0)


class Group:
    """Machines of one kind. | joins alternatives: tci.cpu | tci.mps."""

    def __init__(self, code: str, has: Callable[[Machine], bool]) -> None:
        self.code = code  # as a test writes it
        self.has = has

    def __or__(self, other: Group) -> Group:
        if not isinstance(other, Group):
            return NotImplemented
        either = f"{self.code} | {other.code}"
        return Group(either, lambda machine: self.has(machine) or other.has(machine))


class _Accelerator(Group):
    """Machines with one accelerator's GPUs. Each of its DEVICES is an attribute:
    tci.cuda.h100 is the machines with H100s."""

    def __init__(self, name: str) -> None:
        super().__init__(f"tci.{name}", lambda machine: machine.accelerator == name)
        self._name = name

    def __getattr__(self, device: str) -> Group:
        if device.startswith("_"):  # not a device: copy and pickle look for these
            raise AttributeError(device)
        names = DEVICES[self._name]
        if device not in names:
            raise AttributeError(f"{self.code} has no {device!r} among {list(names)}")
        named = re.compile(rf"{re.escape(names[device])}\b")  # an A10G is not an A10

        def is_one(machine: Machine) -> bool:
            return self.has(machine) and bool(named.match(machine.device))

        return Group(f"{self.code}.{device}", is_one)


class _Cuda(_Accelerator):
    def __call__(self, *, min_capability: str) -> Group:
        """Its machines whose GPUs have this compute capability or a newer one:
        tci.cuda(min_capability="9.0") is Hopper and later. Only CUDA takes it:
        through the same call, ROCm reports numbers of its own."""
        if not re.fullmatch(r"\d+\.\d+", min_capability):
            raise ValueError(f'min_capability is like "9.0", not {min_capability!r}')
        floor = tuple(map(int, min_capability.split(".")))

        def is_newer(machine: Machine) -> bool:
            return self.has(machine) and machine.capability >= floor

        return Group(f'{self.code}(min_capability="{min_capability}")', is_newer)


cpu = Group("tci.cpu", lambda machine: not machine.accelerator)  # no GPU
gpu = Group("tci.gpu", lambda machine: bool(machine.accelerator))
cuda = _Cuda("cuda")
rocm = _Accelerator("rocm")
xpu = _Accelerator("xpu")
mps = _Accelerator("mps")
# Named groups: a feature that GPUs of several vendors have
_fp8 = cuda(min_capability="8.9") | rocm.mi300x | rocm.mi350x
FP8_GPU = Group("tci.FP8_GPU", _fp8.has)

_COMPARE: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}


@dataclass(frozen=True)
class Comparison:
    """What comparing a quantity makes, for @tci.needs: tci.gpus >= 2."""

    code: str  # as a test writes it
    holds: Callable[[Machine], bool]

    def __bool__(self) -> bool:
        # Python reads 1 <= tci.gpus <= 2 as two comparisons joined by "and", and
        # would keep only the second
        raise TypeError("one comparison at a time: tci.gpus >= 1, tci.gpus <= 2")


class _Quantity:
    """Something a machine has an amount of. Comparing it makes a Comparison."""

    def __init__(
        self,
        name: str,
        of: Callable[[Machine], float | None],
        amount: Callable[[Any], float],
    ) -> None:
        self.name = name
        self.of = of  # the machine's amount; None if nothing reports it
        self.amount = amount  # the amount a test wrote, or an error if it is none

    def _compare(self, symbol: str, value: Any) -> Comparison:
        wanted = self.amount(value)

        def holds(machine: Machine) -> bool:
            have = self.of(machine)
            # What nothing reports can't be judged, so it takes no machine away
            return have is None or _COMPARE[symbol](have, wanted)

        return Comparison(f"tci.{self.name} {symbol} {json.dumps(value)}", holds)

    def __ge__(self, value: Any) -> Comparison:
        return self._compare(">=", value)

    def __le__(self, value: Any) -> Comparison:
        return self._compare("<=", value)

    def __eq__(self, value: object) -> Comparison:
        return self._compare("==", value)

    def __ne__(self, value: object) -> Comparison:
        return self._compare("!=", value)

    def __gt__(self, value: Any) -> Comparison:
        return self._compare(">", value)

    def __lt__(self, value: Any) -> Comparison:
        return self._compare("<", value)


def _whole(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"tci.gpus compares with a whole number, not {value!r}")
    return value


def _gib(value: Any) -> int:
    """32 from "32GB": GiB, as largeTensorTest reads the same string."""
    if not isinstance(value, str) or not re.fullmatch(r"\d+GB", value):
        raise ValueError(f'tci.gpu_memory compares with a size like "32GB": {value!r}')
    return int(value[:-2])


def _memory(machine: Machine) -> float | None:
    return machine.gpu_memory if machine.gpus else 0


gpus = _Quantity("gpus", lambda machine: machine.gpus, _whole)
# All of the memory on each GPU, as its driver reports it: an L4 has 22.5 GiB. How
# much of it is free when the test runs is still for largeTensorTest to check
gpu_memory = _Quantity("gpu_memory", _memory, _gib)


@dataclass(frozen=True)
class Constraint:
    code: str  # the decorator, as a test writes it
    met_by: Callable[[Machine], bool]


def _written(values: tuple[Any, ...], kind: type, takes: str) -> str:
    """The arguments as a test writes them, each checked to be of the kind."""
    if not values or not all(isinstance(value, kind) for value in values):
        raise TypeError(takes)
    return ", ".join(value.code for value in values)


def _constrain(code: str, met_by: Callable[[Machine], bool]) -> Callable[[T], T]:
    def decorate(test: T) -> T:
        # Before those under it, to read as written; vars leaves out a base's
        setattr(test, _ATTR, (Constraint(code, met_by), *vars(test).get(_ATTR, ())))
        return test

    return decorate


def only(*groups: Group) -> Callable[[T], T]:
    """The test runs only on machines in one of the groups."""
    written = _written(groups, Group, "only() takes groups, such as tci.cuda")
    return _constrain(f"@tci.only({written})", reduce(operator.or_, groups).has)


def exclude(*groups: Group, reason: str) -> Callable[[T], T]:
    """The test doesn't run on machines in any of the groups. The reason is for
    whoever finds the test left out."""
    written = _written(groups, Group, "exclude() takes groups, such as tci.rocm")
    if not reason.strip():
        raise ValueError("exclude() takes a reason for whoever finds the test left out")
    excluded = reduce(operator.or_, groups).has
    code = f"@tci.exclude({written}, reason={json.dumps(reason)})"
    return _constrain(code, lambda machine: not excluded(machine))


def needs(*comparisons: Comparison, on: Group | None = None) -> Callable[[T], T]:
    """The machine passes each comparison, such as tci.gpus >= 2. on= is the if:
    with on=tci.gpu, only machines that have a GPU are asked."""
    takes = "needs() takes comparisons, such as tci.gpus >= 2"
    written = _written(comparisons, Comparison, takes)
    if on is not None:
        written += ", on=" + _written((on,), Group, "on= takes a group like tci.gpu")

    def met_by(machine: Machine) -> bool:
        asked = on is None or on.has(machine)
        return not asked or all(c.holds(machine) for c in comparisons)

    return _constrain(f"@tci.needs({written})", met_by)


def constraints(cls: type | None, test: object) -> list[Constraint]:
    """Every constraint on a test: those of its class's bases, of its class, then
    its own, each in the order written."""
    levels = reversed(cls.__mro__) if cls else ()
    found = [c for level in levels for c in vars(level).get(_ATTR, ())]
    return [*found, *getattr(test, _ATTR, ())]
