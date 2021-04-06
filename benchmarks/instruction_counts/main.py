"""Basic runner for the instruction count microbenchmarks.

The contents of this file are placeholders, and will be replaced by more
expressive and robust components (e.g. better runner and result display
components) in future iterations. However this allows us to excercise the
underlying benchmark generation infrastructure in the mean time.
"""

from core.expand import materialize
from definitions.standard import BENCHMARKS
from execution.runner import Runner
from execution.work import WorkOrder


def main() -> None:
    work_orders = tuple(
        WorkOrder(label, autolabels, timer_args, timeout=600, retries=2)
        for label, autolabels, timer_args in materialize(BENCHMARKS)
    )

    results = Runner(work_orders).run()
    for work_order in work_orders:
        print(work_order.label, work_order.autolabels, work_order.timer_args.num_threads, results[work_order].instructions)


if __name__ == "__main__":
    main()
