import itertools as it
import os
import subprocess
import textwrap
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

from core.api import AutogradMode, AutoLabels, RuntimeMode
from core.types import Label
from core.utils import unpack
from definitions.ad_hoc import ADHOC_BENCHMARKS
from definitions.standard import BENCHMARKS
from execution.runner import Runner
from execution.work import PYTHON_CMD, WorkOrder
from frontend.display import ResultType, ValueType
from worker.main import WorkerTimerArgs

if TYPE_CHECKING:
    # See core.api for an explanation why this is necessary.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language

from torch.utils.benchmark.utils.historic.patch import backport, clean_backport


_BACKTEST_EXCLUDE: Set[Label] = {
    ('Pointwise', 'Math', 'add', 'Tensor-Tensor (type promotion)'),
    ('Reduction', 'Variance'),
    ('Indexing',),  # All indexing ops
    ('Mesoscale', 'MatMul-Bias-ReLU'),
    ('AutoGrad', 'simple'),
    ('AutoGrad', 'intermediate'),
}


def patch_benchmark_utils(source_cmd: Optional[str], clean_only: bool) -> None:
    if source_cmd is None:
        return None

    cmd = f'{source_cmd} && {PYTHON_CMD} -c "import torch;print(torch.__file__)"'
    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        executable="/bin/bash",
    )

    if proc.returncode:
        raise OSError(f"Patch command failed:\n{proc.stderr}")

    torch_init_file: str = proc.stdout.strip()
    assert os.path.exists(torch_init_file), f"`{torch_init_file}` does not exist."
    assert os.path.split(torch_init_file)[1] == "__init__.py"
    torch_path = os.path.split(torch_init_file)[0]

    if clean_only:
        clean_backport(torch_path)
        return None

    # `backport` will go through its own round of checks.
    backport(torch_path)


def _make_sentry(source_cmd: Optional[str]) -> WorkOrder:
    """Known stable tasks which are injected to test measurement stability."""
    timer_args = WorkerTimerArgs(
        stmt="""
            auto x = torch::ones({4, 4});
            auto z = x - y;
        """,
        setup="auto y = torch::ones({4, 4});",
        global_setup=None,
        num_threads=1,
        language=Language.CPP,
    )

    return WorkOrder(
        label=("Impl", "Sentry"),
        auto_labels=AutoLabels(
            RuntimeMode.EAGER,
            AutogradMode.FORWARD,
            language=Language.CPP
        ),
        timer_args=timer_args,
        source_cmd=source_cmd,
        timeout=240.0,
        retries=2,
    )


def collect(
    source_cmds: Tuple[Optional[str], ...] = (None,),
    ad_hoc: bool = False,
    no_cpp: bool = False,
    backtesting: bool = False,
) -> Tuple[ResultType, ...]:
    all_work_items: List[WorkOrder] = []
    work_items_by_source_cmd: List[Tuple[WorkOrder, ...]] = []

    # Set up normal benchmarks
    benchmarks = ADHOC_BENCHMARKS if ad_hoc else BENCHMARKS
    for label, auto_labels, timer_args in unpack(benchmarks):
        if no_cpp and auto_labels.language == Language.CPP:
            continue

        if backtesting:
            if auto_labels.runtime == RuntimeMode.JIT:
                continue

            if any(label[:i + 1] in _BACKTEST_EXCLUDE for i in range(len(label))):
                continue

        orders: Tuple[WorkOrder, ...] = tuple(
            WorkOrder(
                label=label,
                auto_labels=auto_labels,
                timer_args=timer_args,
                source_cmd=source_cmd,
                timeout=300.0,
                retries=2,
            )
            for source_cmd in source_cmds
        )
        work_items_by_source_cmd.append(orders)
        all_work_items.extend(orders)

    # Set up sentry measurements for warnings.
    sentry_work_items: List[Tuple[WorkOrder, ...]] = [
        tuple(_make_sentry(source_cmd) for _ in range(3))
        for source_cmd in source_cmds
    ]
    if not no_cpp:
        all_work_items = list(it.chain(*sentry_work_items)) + all_work_items

    # Collect measurements.
    results = Runner(work_items=tuple(all_work_items)).run()

    # Warn if there is ANY variation in instruction counts. While Python has
    # some jitter, C++ should be truly detministic.
    for source_cmd, work_items in zip(source_cmds, sentry_work_items):
        if no_cpp:
            continue
        sentry_results = [results[w].instructions.counts() for w in work_items]
        if len(set(sentry_results)) > 1:
            print(textwrap.dedent(f"""
                WARNING: measurements are unstable. (source cmd: `{source_cmd}`)
                    Three C++ sentries were run which should have been completely
                    deterministic, but instead resulted in the following counts:
                      {sentry_results}
            """))

    # Organize normal benchmark results.
    output: List[ResultType] = []
    for work_items in zip(*work_items_by_source_cmd):
        output_i: List[Tuple[Label, int, AutoLabels, ValueType]] = []
        for w in work_items:
            r = results[w]
            output_i.append((
                w.label,
                w.timer_args.num_threads,
                w.auto_labels,
                (r.instructions, r.wall_time)
            ))
        output.append(tuple(output_i))

    return tuple(output)
