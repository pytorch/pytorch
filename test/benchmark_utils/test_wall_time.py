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

    def __init__(self, stmt: str, setup: str):
        self._stmt: str = stmt
        self._setup: str = setup

        self._compiled_module_created: bool = False

        self._state = random.Random(self._seed)
        self._mean_cost: float = {k: v for k, v in self._function_costs}[stmt]
        self._last_measurement: typing.Optional[float] = None

    def _sample(self, mean: float, std_over_mean: float) -> float:
        return self._state.normalvariate(mean, std_over_mean * mean)

    def run(self, snippet: str) -> None:
        if self.is_create_compiled_module(snippet):
            self._compiled_module_created = True
            return

        n_iter = self.is_run_cmd(snippet)
        if n_iter is not None:
            assert self._compiled_module_created

            self._last_measurement = sum([
                # First timer invocation
                self._sample(self._timer_cost, self._timer_noise_level),

                # Stmt body
                self._sample(self._mean_cost * n_iter, self._function_noise_level),

                # Second timer invocation
                self._sample(self._timer_cost, self._timer_noise_level),
            ])

        else:
            raise NotImplementedError(f"Unknown snippet:\n{snippet}")

    def store(self, name: str, value: typing.Any, *, in_memory: bool = False) -> None:
        raise NotImplementedError(f"store: {name} = {value}")

    def load(self, name: str) -> typing.Any:
        if name == "_run_in_worker_result":
            assert self._last_measurement is not None

            # (Time measurement, should_cuda_sync)
            return (self._last_measurement, False)

        raise NotImplementedError(f"load: {name}")

    @property
    def in_process(self) -> bool:
        return True

    def is_create_compiled_module(self, snippet: str) -> bool:
        # Borrowed from CPython timeit.
        def reindent(src: str, indent: int) -> str:
            return src.replace("\n", "\n" + " " * indent)

        # We don't need to check the entire definition, as that would be
        # tedious and brittle
        template = textwrap.dedent(f"""
            class CompiledTimerModule:

                @staticmethod
                def call(n_iter: int) -> None:
                    {reindent(self._setup, 20)}

                    for _ in range(n_iter):
                        {reindent(self._stmt, 24)}
                        pass
        """)

        # We don't bother with regex here; it is enough to check that a
        # specific substring appears.
        return template in snippet

    @staticmethod
    def is_run_cmd(snippet: str) -> typing.Optional[int]:
        template = """
            def _run_in_worker_f():
                ...
                n_iter = marshal.loads(bytes.fromhex('{hex_capture}'))  # {num_capture}
                ...
                with runtime_utils.set_torch_threads(num_threads):
                    ...
                    return jit_template.get().measure_wall_time(
                        n_iter=n_iter,
                        ...
                    ), should_cuda_sync
            ...
            _run_in_worker_result = _run_in_worker_f()
        """

        # Raw regex can be hard to read, particularly the multi line skip
        # pattern. So instead we use some syntactic sugar to start with a human
        # readable template and convert it to a regex pattern.
        pattern = re.escape(textwrap.dedent(template).strip()) \
            .replace(r"\{hex_capture\}", "{hex_capture}") \
            .replace(r"\{num_capture\}", "{num_capture}") \
            .format(hex_capture="([0-9a-f]+)", num_capture="([0-9]+)") \
            .replace(r"\.\.\.", r".*(?:[\n\r]^.*$)*")

        n_iter: typing.Optional[int] = None
        match = re.search(pattern, snippet, re.MULTILINE)
        if match:
            n_iter = marshal.loads(bytes.fromhex(match.groups()[0]))
            assert isinstance(n_iter, int)
            assert n_iter == int(match.groups()[1]), f"{n_iter} != {match.groups()[1]}"

        return n_iter


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
            worker=MockWorker(stmt=work_spec.stmt, setup=work_spec.setup)
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
