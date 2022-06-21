import inspect
from typing import Callable, List

import torch.fx as fx
from .pass_manager import PassManager


class PassPipelineManager:
    """
    Construct a PassPipelineManager.

    Collects a list of pass managers which are run in order.

    Args:
        pass_managers (Optional[List[PassManager]]): list of pass managers
    """

    pass_managers: List[PassManager] = []

    def __init__(
        self,
        pass_managers=None,
        run_checks_after_each_pass: bool = False,
    ):
        if pass_managers:
            self.pass_managers = pass_managers
        self.run_checks_after_each_pass = run_checks_after_each_pass

    def add_checks(self, check: Callable) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        sig = inspect.signature(check)

        if len(list(sig.parameters.values())) != 1:
            raise TypeError("PassManager check function should only take in one variable, a graph_module")

        setattr(self, "check", check)  # noqa: B010

    def check(self, graph_module: fx.GraphModule) -> None:
        pass

    def __call__(self, graph_module: fx.GraphModule) -> None:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module to ensure that it still maintains the same required
        invariants.
        """

        self.check(graph_module)

        for pm in self.pass_managers:
            pm(graph_module)

            if self.run_checks_after_each_pass:
                self.check(graph_module)

        graph_module.recompile()
