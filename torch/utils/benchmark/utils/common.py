"""Base shared classes and utilities."""

import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid

import torch


__all__ = ["TaskSpec", "Measurement", "select_unit", "unit_to_english", "trim_sigfig", "ordered_unique", "set_torch_threads"]


_MAX_SIGNIFICANT_FIGURES = 4
_MIN_CONFIDENCE_INTERVAL = 25e-9  # 25 ns

# Measurement will include a warning if the distribution is suspect. All
# runs are expected to have some variation; these parameters set the
# thresholds.
_IQR_WARN_THRESHOLD = 0.1
_IQR_GROSS_WARN_THRESHOLD = 0.25


@dataclasses.dataclass(init=True, repr=False, eq=True, frozen=True)
class TaskSpec:
    """Container for information used to define a Timer. (except globals)"""
    stmt: str
    setup: str
    global_setup: str = ""
    label: Optional[str] = None
    sub_label: Optional[str] = None
    description: Optional[str] = None
    env: Optional[str] = None
    num_threads: int = 1

    @property
    def title(self) -> str:
        """Best effort attempt at a string label for the measurement."""
        if self.label is not None:
            return self.label + (f": {self.sub_label}" if self.sub_label else "")
        elif "\n" not in self.stmt:
            return self.stmt + (f": {self.sub_label}" if self.sub_label else "")
        return (
            f"stmt:{f' ({self.sub_label})' if self.sub_label else ''}\n"
            f"{textwrap.indent(self.stmt, '  ')}"
        )

    def setup_str(self) -> str:
        return (
            "" if (self.setup == "pass" or not self.setup)
            else f"setup:\n{textwrap.indent(self.setup, '  ')}" if "\n" in self.setup
            else f"setup: {self.setup}"
        )

    def summarize(self) -> str:
        """Build TaskSpec portion of repr string for other containers."""
        sections = [
            self.title,
            self.description or "",
            self.setup_str(),
        ]
        return "\n".join([f"{i}\n" if "\n" in i else i for i in sections if i])

_TASKSPEC_FIELDS = tuple(i.name for i in dataclasses.fields(TaskSpec))


@dataclasses.dataclass(init=True, repr=False)
class Measurement:
    """The result of a Timer measurement.

    This class stores one or more measurements of a given statement. It is
    serializable and provides several convenience methods
    (including a detailed __repr__) for downstream consumers.
    """
    number_per_run: int
    raw_times: List[float]
    task_spec: TaskSpec
    metadata: Optional[Dict[Any, Any]] = None  # Reserved for user payloads.

    def __post_init__(self) -> None:
        self._sorted_times: Tuple[float, ...] = ()
        self._warnings: Tuple[str, ...] = ()
        self._median: float = -1.0
        self._mean: float = -1.0
        self._p25: float = -1.0
        self._p75: float = -1.0

    def __getattr__(self, name: str) -> Any:
        # Forward TaskSpec fields for convenience.
        if name in _TASKSPEC_FIELDS:
            return getattr(self.task_spec, name)
        return super().__getattribute__(name)

    # =========================================================================
    # == Convenience methods for statistics ===================================
    # =========================================================================
    #
    # These methods use raw time divided by number_per_run; this is an
    # extrapolation and hides the fact that different number_per_run will
    # result in different amortization of overheads, however if Timer has
    # selected an appropriate number_per_run then this is a non-issue, and
    # forcing users to handle that division would result in a poor experience.
    @property
    def times(self) -> List[float]:
        return [t / self.number_per_run for t in self.raw_times]

    @property
    def median(self) -> float:
        self._lazy_init()
        return self._median

    @property
    def mean(self) -> float:
        self._lazy_init()
        return self._mean

    @property
    def iqr(self) -> float:
        self._lazy_init()
        return self._p75 - self._p25

    @property
    def significant_figures(self) -> int:
        """Approximate significant figure estimate.

        This property is intended to give a convenient way to estimate the
        precision of a measurement. It only uses the interquartile region to
        estimate statistics to try to mitigate skew from the tails, and
        uses a static z value of 1.645 since it is not expected to be used
        for small values of `n`, so z can approximate `t`.

        The significant figure estimation used in conjunction with the
        `trim_sigfig` method to provide a more human interpretable data
        summary. __repr__ does not use this method; it simply displays raw
        values. Significant figure estimation is intended for `Compare`.
        """
        self._lazy_init()
        n_total = len(self._sorted_times)
        lower_bound = int(n_total // 4)
        upper_bound = int(torch.tensor(3 * n_total / 4).ceil())
        interquartile_points: Tuple[float, ...] = self._sorted_times[lower_bound:upper_bound]
        std = torch.tensor(interquartile_points).std(unbiased=False).item()
        sqrt_n = torch.tensor(len(interquartile_points)).sqrt().item()

        # Rough estimates. These are by no means statistically rigorous.
        confidence_interval = max(1.645 * std / sqrt_n, _MIN_CONFIDENCE_INTERVAL)
        relative_ci = torch.tensor(self._median / confidence_interval).log10().item()
        num_significant_figures = int(torch.tensor(relative_ci).floor())
        return min(max(num_significant_figures, 1), _MAX_SIGNIFICANT_FIGURES)

    @property
    def has_warnings(self) -> bool:
        self._lazy_init()
        return bool(self._warnings)

    def _lazy_init(self) -> None:
        if self.raw_times and not self._sorted_times:
            self._sorted_times = tuple(sorted(self.times))
            _sorted_times = torch.tensor(self._sorted_times, dtype=torch.float64)
            self._median = _sorted_times.quantile(.5).item()
            self._mean = _sorted_times.mean().item()
            self._p25 = _sorted_times.quantile(.25).item()
            self._p75 = _sorted_times.quantile(.75).item()

            def add_warning(msg: str) -> None:
                rel_iqr = self.iqr / self.median * 100
                self._warnings += (
                    f"  WARNING: Interquartile range is {rel_iqr:.1f}% "
                    f"of the median measurement.\n           {msg}",
                )

            if not self.meets_confidence(_IQR_GROSS_WARN_THRESHOLD):
                add_warning("This suggests significant environmental influence.")
            elif not self.meets_confidence(_IQR_WARN_THRESHOLD):
                add_warning("This could indicate system fluctuation.")


    def meets_confidence(self, threshold: float = _IQR_WARN_THRESHOLD) -> bool:
        return self.iqr / self.median < threshold

    @property
    def title(self) -> str:
        return self.task_spec.title

    @property
    def env(self) -> str:
        return (
            "Unspecified env" if self.taskspec.env is None
            else cast(str, self.taskspec.env)
        )

    @property
    def as_row_name(self) -> str:
        return self.sub_label or self.stmt or "[Unknown]"

    def __repr__(self) -> str:
        """
        Example repr:
            <utils.common.Measurement object at 0x7f395b6ac110>
              Broadcasting add (4x8)
              Median: 5.73 us
              IQR:    2.25 us (4.01 to 6.26)
              372 measurements, 100 runs per measurement, 1 thread
              WARNING: Interquartile range is 39.4% of the median measurement.
                       This suggests significant environmental influence.
        """
        self._lazy_init()
        skip_line, newline = "MEASUREMENT_REPR_SKIP_LINE", "\n"
        n = len(self._sorted_times)
        time_unit, time_scale = select_unit(self._median)
        iqr_filter = '' if n >= 4 else skip_line

        repr_str = f"""
{super().__repr__()}
{self.task_spec.summarize()}
  {'Median: ' if n > 1 else ''}{self._median / time_scale:.2f} {time_unit}
  {iqr_filter}IQR:    {self.iqr / time_scale:.2f} {time_unit} ({self._p25 / time_scale:.2f} to {self._p75 / time_scale:.2f})
  {n} measurement{'s' if n > 1 else ''}, {self.number_per_run} runs {'per measurement,' if n > 1 else ','} {self.num_threads} thread{'s' if self.num_threads > 1 else ''}
{newline.join(self._warnings)}""".strip()  # noqa: B950

        return "\n".join(l for l in repr_str.splitlines(keepends=False) if skip_line not in l)

    @staticmethod
    def merge(measurements: Iterable["Measurement"]) -> List["Measurement"]:
        """Convenience method for merging replicates.

        Merge will extrapolate times to `number_per_run=1` and will not
        transfer any metadata. (Since it might differ between replicates)
        """
        grouped_measurements: DefaultDict[TaskSpec, List["Measurement"]] = collections.defaultdict(list)
        for m in measurements:
            grouped_measurements[m.task_spec].append(m)

        def merge_group(task_spec: TaskSpec, group: List["Measurement"]) -> "Measurement":
            times: List[float] = []
            for m in group:
                # Different measurements could have different `number_per_run`,
                # so we call `.times` which normalizes the results.
                times.extend(m.times)

            return Measurement(
                number_per_run=1,
                raw_times=times,
                task_spec=task_spec,
                metadata=None,
            )

        return [merge_group(t, g) for t, g in grouped_measurements.items()]


def select_unit(t: float) -> Tuple[str, float]:
    """Determine how to scale times for O(1) magnitude.

    This utility is used to format numbers for human consumption.
    """
    time_unit = {-3: "ns", -2: "us", -1: "ms"}.get(int(torch.tensor(t).log10().item() // 3), "s")
    time_scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[time_unit]
    return time_unit, time_scale


def unit_to_english(u: str) -> str:
    return {
        "ns": "nanosecond",
        "us": "microsecond",
        "ms": "millisecond",
        "s": "second",
    }[u]


def trim_sigfig(x: float, n: int) -> float:
    """Trim `x` to `n` significant figures. (e.g. 3.14159, 2 -> 3.10000)"""
    assert n == int(n)
    magnitude = int(torch.tensor(x).abs().log10().ceil().item())
    scale = 10 ** (magnitude - n)
    return float(torch.tensor(x / scale).round() * scale)


def ordered_unique(elements: Iterable[Any]) -> List[Any]:
    return list(collections.OrderedDict({i: None for i in elements}).keys())


@contextlib.contextmanager
def set_torch_threads(n: int) -> Iterator[None]:
    prior_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(n)
        yield
    finally:
        torch.set_num_threads(prior_num_threads)


def _make_temp_dir(prefix: Optional[str] = None, gc_dev_shm: bool = False) -> str:
    """Create a temporary directory. The caller is responsible for cleanup.

    This function is conceptually similar to `tempfile.mkdtemp`, but with
    the key additional feature that it will use shared memory if the
    `BENCHMARK_USE_DEV_SHM` environment variable is set. This is an
    implementation detail, but an important one for cases where many Callgrind
    measurements are collected at once. (Such as when collecting
    microbenchmarks.)

    This is an internal utility, and is exported solely so that microbenchmarks
    can reuse the util.
    """
    use_dev_shm: bool = (os.getenv("BENCHMARK_USE_DEV_SHM") or "").lower() in ("1", "true")
    if use_dev_shm:
        root = "/dev/shm/pytorch_benchmark_utils"
        assert os.name == "posix", f"tmpfs (/dev/shm) is POSIX only, current platform is {os.name}"
        assert os.path.exists("/dev/shm"), "This system does not appear to support tmpfs (/dev/shm)."
        os.makedirs(root, exist_ok=True)

        # Because we're working in shared memory, it is more important than
        # usual to clean up ALL intermediate files. However we don't want every
        # worker to walk over all outstanding directories, so instead we only
        # check when we are sure that it won't lead to contention.
        if gc_dev_shm:
            for i in os.listdir(root):
                owner_file = os.path.join(root, i, "owner.pid")
                if not os.path.exists(owner_file):
                    continue

                with open(owner_file) as f:
                    owner_pid = int(f.read())

                if owner_pid == os.getpid():
                    continue

                try:
                    # https://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid-in-python
                    os.kill(owner_pid, 0)

                except OSError:
                    print(f"Detected that {os.path.join(root, i)} was orphaned in shared memory. Cleaning up.")
                    shutil.rmtree(os.path.join(root, i))

    else:
        root = tempfile.gettempdir()

    # We include the time so names sort by creation time, and add a UUID
    # to ensure we don't collide.
    name = f"{prefix or tempfile.gettempprefix()}__{int(time.time())}__{uuid.uuid4()}"
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=False)

    if use_dev_shm:
        with open(os.path.join(path, "owner.pid"), "w") as f:
            f.write(str(os.getpid()))

    return path
