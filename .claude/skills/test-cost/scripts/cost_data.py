"""Pure data model for the test-cost report.

Maps runner labels to hardware classes, invoking files to test files and owners,
and aggregates the per-day ClickHouse rows into a Report. Nothing here touches
the network or the clock, so the same rows always give the same Report.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


HW_CLASSES = (
    "CPU x86",
    "CPU arm64",
    "Windows",
    "macOS",
    "T4",
    "A10G",
    "L4",
    "A100",
    "H100",
    "B200",
    "ROCm",
    "XPU",
    "TPU",
    "s390x",
    "unknown",
)
# Chart families in stack order; the palette slot follows this order.
FAMILIES = (
    "CPU x86",
    "NVIDIA GPU",
    "ROCm",
    "Other accelerator",
    "CPU arm64",
    "Other CPU",
)
HW_FAMILY = {
    "CPU x86": "CPU x86",
    "CPU arm64": "CPU arm64",
    "Windows": "Other CPU",
    "macOS": "Other CPU",
    "T4": "NVIDIA GPU",
    "A10G": "NVIDIA GPU",
    "L4": "NVIDIA GPU",
    "A100": "NVIDIA GPU",
    "H100": "NVIDIA GPU",
    "B200": "NVIDIA GPU",
    "ROCm": "ROCm",
    "XPU": "Other accelerator",
    "TPU": "Other accelerator",
    "s390x": "Other CPU",
    "unknown": "Other accelerator",
}
ACCELERATORS = frozenset(
    {"T4", "A10G", "L4", "A100", "H100", "B200", "ROCm", "XPU", "TPU"}
)
TRIGGERS = ("main", "pr", "scheduled", "other")

# First match wins. GPU tokens come before OS and CPU tokens so windows.g4dn.xlarge
# is T4, and the accelerator guard keeps an unrecognised GPU fleet out of the CPU
# classes. Label grammar: .github/arc.yaml (ARC) plus the legacy EC2 family names.
HW_PATTERNS = tuple(
    (re.compile(pattern), hw_class)
    for pattern, hw_class in (
        (r"(^|[.-])h100([.-]|$)|[.-]p5[.-]", "H100"),
        (r"(^|[.-])b200([.-]|$)", "B200"),
        (r"(^|[.-])a100([.-]|$)|[.-]p4de?[.-]", "A100"),
        (r"(^|[.-])a10g([.-]|$)|[.-]g5[.-]", "A10G"),
        (r"(^|[.-])l4([.-]|$)|[.-]g6[.-]", "L4"),
        (r"(^|[.-])t4([.-]|$)|[.-]g4dn[.-]", "T4"),
        (r"rocm|[.-]mi\d{3}|gfx\d+|rx7900", "ROCm"),
        (r"(^|[.-])xpu([.-]|$)", "XPU"),
        (r"tpu", "TPU"),
        (r"s390x", "s390x"),
        (r"nvidia|gpu|dgx|hpu|gaudi|l40s|g6e|h200", "unknown"),
        (r"^win", "Windows"),
        (r"^macos", "macOS"),
        (r"^(mt|lf)-w-", "Windows"),
        (r"^(mt|lf)-m-", "macOS"),
        (r"arm64|aarch64|graviton|[.-](m7g|m8g|c7g|t4g|r7g)[.-]", "CPU arm64"),
        (r"x86|^linux\.|^lf\.linux\.|^ubuntu-|xlarge", "CPU x86"),
    )
)

# invoking_file is the test path with "/" replaced by "." and ".py" dropped (see
# sanitize_test_filename in torch/testing/_internal/common_utils.py). A few
# run_test.py entries execute a differently named file.
INVOKING_ALIASES = {
    "test_cpp_extensions_aot_ninja": "test_cpp_extensions_aot",
    "test_cpp_extensions_aot_no_ninja": "test_cpp_extensions_aot",
    "test_custom_backend": "custom_backend.test_custom_backend",
}
LEAKED_PREFIX = ".__w.pytorch.pytorch.test."
OWNER_PREFIX = "# Owner(s): "
LABEL_PREFIX_RE = re.compile(r"^(module|oncall):\s*")


def ratio(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def hardware_class(label: str) -> str:
    for pattern, hw_class in HW_PATTERNS:
        if pattern.search(label):
            return hw_class
    return "unknown"


def invoking_file_to_path(name: str, test_dir: Path) -> str | None:
    module = name.replace("/", ".").removeprefix(LEAKED_PREFIX)
    module = INVOKING_ALIASES.get(module, module)
    rel = module.replace(".", "/") + ".py"
    return f"test/{rel}" if (test_dir / rel).is_file() else None


def read_owner(path: Path) -> str:
    """First label of the ``# Owner(s): [...]`` header without its module:/oncall: prefix."""
    with path.open(encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            if not line.startswith(OWNER_PREFIX):
                continue
            try:
                labels = ast.literal_eval(line[len(OWNER_PREFIX) :].strip())
            except (ValueError, SyntaxError):
                return "no-header"
            if (
                not isinstance(labels, list)
                or not labels
                or not isinstance(labels[0], str)
            ):
                return "no-header"
            return LABEL_PREFIX_RE.sub("", labels[0].strip()).strip() or "no-header"
    return "no-header"


@dataclass(frozen=True)
class Window:
    start: date
    end: date  # exclusive

    @property
    def last(self) -> date:
        return self.end - timedelta(days=1)

    @property
    def days(self) -> list[date]:
        return [
            self.start + timedelta(days=i) for i in range((self.end - self.start).days)
        ]


@dataclass(frozen=True)
class JobRow:
    runner: str
    workflow: str
    trigger: str
    conclusion: str
    jobs: int
    wall_s: int


@dataclass(frozen=True)
class FileRow:
    """One canonical invoking file, retaining its original invoking names."""

    runner: str
    workflow: str
    trigger: str
    invoking_file: str
    test_s: float
    attr_s: float
    test_rows: int
    jobs: int
    invoking_files: list[str]


@dataclass
class DayData:
    day: date
    jobs: list[JobRow]
    files: list[FileRow]
    settled: bool
    cached: bool


@dataclass
class Agg:
    """Attributed hours for one owner (members are file paths) or one file (members are invoking names)."""

    key: str
    owner: str
    by_class: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(HW_CLASSES, 0.0)
    )
    by_trigger: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(TRIGGERS, 0.0)
    )
    members: set[str] = field(default_factory=set)
    jobs: int = 0
    test_h: float = 0.0

    @property
    def hours(self) -> float:
        return sum(self.by_class.values())


@dataclass
class LabelAgg:
    label: str
    hw_class: str
    jobs: int = 0
    wall_h: float = 0.0
    attr_h: float = 0.0


@dataclass
class WorkflowAgg:
    workflow: str
    jobs: int = 0
    wall_h: float = 0.0
    attr_h: float = 0.0


@dataclass
class DayAgg:
    day: date
    settled: bool
    cached: bool
    jobs: int = 0
    wall_h: float = 0.0
    attr_h: float = 0.0


@dataclass
class UnmappedAgg:
    invoking_file: str
    attr_h: float = 0.0
    jobs: int = 0


@dataclass
class VerifyRow:
    job_id: int
    run_id: int
    ch_label: str
    ch_wall_s: int
    gh_label: str = ""
    gh_wall_s: int | None = None
    status: str = "ERROR"  # OK, MISMATCH or ERROR
    note: str = ""


@dataclass(frozen=True)
class Meta:
    window: Window
    query_hash: str
    checkout: str
    command: str


@dataclass
class Report:
    meta: Meta
    days: list[DayAgg]
    owners: list[Agg]
    files: list[Agg]
    labels: list[LabelAgg]
    workflows: list[WorkflowAgg]
    unmapped: list[UnmappedAgg]
    class_wall_h: dict[str, float]
    class_attr_h: dict[str, float]
    trigger_wall_h: dict[str, float]
    trigger_attr_h: dict[str, float]
    total_jobs: int
    warnings: list[str]
    verify: list[VerifyRow] = field(default_factory=list)
    verify_requested: int = 0

    @property
    def wall_h(self) -> float:
        return sum(self.class_wall_h.values())

    @property
    def attr_h(self) -> float:
        return sum(self.class_attr_h.values())

    @property
    def coverage(self) -> float:
        return ratio(self.attr_h, self.wall_h)

    @property
    def accelerator_share(self) -> float:
        accel = sum(h for c, h in self.class_wall_h.items() if c in ACCELERATORS)
        return ratio(accel, self.wall_h)


def aggregate(days: list[DayData], repo_root: Path, meta: Meta) -> Report:
    test_dir = repo_root / "test"
    owners: dict[str, Agg] = {}
    files: dict[str, Agg] = {}
    labels: dict[str, LabelAgg] = {}
    workflows: dict[str, WorkflowAgg] = {}
    unmapped: dict[str, UnmappedAgg] = {}
    path_of: dict[str, str | None] = {}
    owner_of: dict[str, str] = {}
    class_wall = dict.fromkeys(HW_CLASSES, 0.0)
    class_attr = dict.fromkeys(HW_CLASSES, 0.0)
    trigger_wall = dict.fromkeys(TRIGGERS, 0.0)
    trigger_attr = dict.fromkeys(TRIGGERS, 0.0)
    day_aggs: list[DayAgg] = []
    warnings: list[str] = []
    for data in days:
        day = DayAgg(data.day, data.settled, data.cached)
        for job in data.jobs:
            hours = job.wall_s / 3600
            hw = hardware_class(job.runner)
            label = labels.setdefault(job.runner, LabelAgg(job.runner, hw))
            label.jobs += job.jobs
            label.wall_h += hours
            workflow = workflows.setdefault(job.workflow, WorkflowAgg(job.workflow))
            workflow.jobs += job.jobs
            workflow.wall_h += hours
            class_wall[hw] += hours
            trigger_wall[job.trigger] += hours
            day.jobs += job.jobs
            day.wall_h += hours
        for row in data.files:
            hours = row.attr_s / 3600
            if hours <= 0:
                continue
            hw = hardware_class(row.runner)
            if row.invoking_file not in path_of:
                path_of[row.invoking_file] = invoking_file_to_path(
                    row.invoking_file, test_dir
                )
            path = path_of[row.invoking_file]
            if path is None:
                owner = "unmapped"
                entry = unmapped.setdefault(
                    row.invoking_file, UnmappedAgg(row.invoking_file)
                )
                entry.attr_h += hours
                entry.jobs += row.jobs
            else:
                if path not in owner_of:
                    owner_of[path] = read_owner(repo_root / path)
                owner = owner_of[path]
                file_agg = files.setdefault(path, Agg(path, owner))
                file_agg.by_class[hw] += hours
                file_agg.by_trigger[row.trigger] += hours
                file_agg.members.update(n.replace("/", ".") for n in row.invoking_files)
                file_agg.jobs += row.jobs
                file_agg.test_h += row.test_s / 3600
            owner_agg = owners.setdefault(owner, Agg(owner, owner))
            owner_agg.by_class[hw] += hours
            owner_agg.by_trigger[row.trigger] += hours
            owner_agg.members.add(path or row.invoking_file)
            owner_agg.jobs += row.jobs
            owner_agg.test_h += row.test_s / 3600
            labels.setdefault(row.runner, LabelAgg(row.runner, hw)).attr_h += hours
            workflow = workflows.setdefault(row.workflow, WorkflowAgg(row.workflow))
            workflow.attr_h += hours
            class_attr[hw] += hours
            trigger_attr[row.trigger] += hours
            day.attr_h += hours
        if day.jobs == 0:
            warnings.append(
                f"{data.day}: no test jobs (ingestion gap or the day is not mirrored yet)"
            )
        elif day.attr_h > day.wall_h * 1.001:
            warnings.append(
                f"{data.day}: attributed {day.attr_h:.1f} h exceeds test-job hours {day.wall_h:.1f} h"
            )
        day_aggs.append(day)
    unknown_h = class_wall["unknown"]
    total_wall = sum(class_wall.values())
    if total_wall and unknown_h > 0.01 * total_wall:
        warnings.append(
            f"unknown hardware class holds {unknown_h / total_wall:.1%} of test-job hours; extend HW_PATTERNS"
        )
    return Report(
        meta=meta,
        days=day_aggs,
        owners=sorted(owners.values(), key=lambda a: (-a.hours, a.key)),
        files=sorted(files.values(), key=lambda a: (-a.hours, a.key)),
        labels=sorted(labels.values(), key=lambda a: (-a.wall_h, a.label)),
        workflows=sorted(workflows.values(), key=lambda a: (-a.wall_h, a.workflow)),
        unmapped=sorted(unmapped.values(), key=lambda a: (-a.attr_h, a.invoking_file)),
        class_wall_h=class_wall,
        class_attr_h=class_attr,
        trigger_wall_h=trigger_wall,
        trigger_attr_h=trigger_attr,
        total_jobs=sum(d.jobs for d in day_aggs),
        warnings=warnings,
    )
