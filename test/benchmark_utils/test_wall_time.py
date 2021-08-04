import marshal
import random
import re
import textwrap
import timeit
import typing

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.benchmark._impl import constants
from torch.utils.benchmark._impl import timer as timer_impl
from torch.utils.benchmark._impl.tasks import wall_time
from torch.utils.benchmark._impl.workers import base as base_worker


class MockWorker(base_worker.WorkerBase):
    """Deterministic worker which can be used to mock out timing algorithms."""

    _seed = 0

    _timer_noise_level = 0.05
    _timer_cost = 100e-9  # 100 ns

    _function_noise_level = 0.05
    _function_costs = (
        # NB: More entries will be added in later PRs.
        ("pass", 8e-9),
    )

    def __init__(self, stmt: str):
        self._state = random.Random(self._seed)
        self._mean_cost: float = {k: v for k, v in self._function_costs}[stmt]
        self._last_measurement: typing.Optional[float] = None

    def _sample(self, mean: float, std_over_mean: float) -> float:
        return self._state.normalvariate(mean, std_over_mean * mean)

    def run(self, snippet: str) -> None:
        pattern = r"""
            def _run_in_worker_f\(\):
                # Deserialize args
                import marshal
                n_iter = marshal\.loads\(bytes.fromhex\('(.+)'\)\)  # ([0-9]+)
        """
        pattern = textwrap.dedent(pattern).strip()
        match = re.search(pattern, snippet, re.MULTILINE)
        if match:
            number = marshal.loads(bytes.fromhex(match.groups()[0]))
            assert number == int(match.groups()[1]), f"{number} != {match.groups()[1]}"

            assert "jit_template.get().measure_wall_time(" in snippet
            assert snippet.endswith("_run_in_worker_result = _run_in_worker_f()")

            self._last_measurement = sum([
                # First timer invocation
                self._sample(self._timer_cost, self._timer_noise_level),

                # Stmt body
                self._sample(self._mean_cost * number, self._function_noise_level),

                # Second timer invocation
                self._sample(self._timer_cost, self._timer_noise_level),
            ])

        # Note:
        #   This is not a very robust emulation of a "proper" worker. We drop
        #   any snippet which doesn't look like a measurement, and don't check
        #   that setup code was run in the first place. The reason for such lax
        #   testing is that the invocations that WallTimeTask sends will change
        #   in later PRs (as various abstractions are added), so it is not
        #   worth doing fine grained checks on the earlier PRs.

    def store(self, name: str, value: typing.Any, *, in_memory: bool = False) -> None:
        raise NotImplementedError(f"store: {name} = {value}")

    def load(self, name: str) -> typing.Any:
        if name == "_run_in_worker_result":
            assert self._last_measurement is not None
            return self._last_measurement

        raise NotImplementedError(f"load: {name}")

    @property
    def in_process(self) -> bool:
        return True


class TestBenchmarkWallTime(TestCase):

    def test_numerics(self) -> None:
        work_spec = constants.WorkSpec(
            stmt="pass",
            setup="pass",
            global_setup="",
            num_threads=1,
            language=constants.Language.PYTHON,
        )
        task = wall_time.TimeitTask(
            work_spec=work_spec,
            timer=timeit.default_timer,
            worker=MockWorker(stmt=work_spec.stmt)
        )

        self.assertEqual(
            [task.timeit(number=100) for _ in range(3)],
            [
                1.0483952930213941e-06,
                0.9528676011634897e-06,
                0.9762990120503803e-06
            ],
        )

    def test_timer(self) -> None:
        timer = timer_impl.Timer("pass")
        assert isinstance(timer.timeit(100), float)

        timer = timer_impl.Timer("torch.ones((1,))", "import torch")
        assert isinstance(timer.timeit(100), float)


if __name__ == "__main__":
    run_tests()
