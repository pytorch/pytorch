from typing import Optional, Tuple, Union

from core.api import AutoLabels, GroupedBenchmark, GroupedSetup, TimerArgs


class _GroupedBenchmarkImpl(GroupedBenchmark):
    """The body will come in a subsequent PR."""

    @staticmethod
    def init_from_stmts(
        py_stmt: Optional[str] = None,
        cpp_stmt: Optional[str] = None,

        # Generic constructor arguments
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, Tuple[int, ...]] = 1,
    ) -> "_GroupedBenchmarkImpl":
        raise NotImplementedError

    @staticmethod
    def init_from_model(
        py_model_setup: Optional[str] = None,
        cpp_model_setup: Optional[str] = None,

        # Generic constructor arguments
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, Tuple[int, ...]] = 1,
    ) -> "_GroupedBenchmarkImpl":
        raise NotImplementedError

    @property
    def ts_model_setup(self) -> Optional[str]:
        raise NotImplementedError

    def flatten(
        self,
        model_path: Optional[str]
    ) -> Tuple[Tuple[AutoLabels, TimerArgs], ...]:
        raise NotImplementedError


GroupedStmts = _GroupedBenchmarkImpl.init_from_stmts
GroupedModules = _GroupedBenchmarkImpl.init_from_model
