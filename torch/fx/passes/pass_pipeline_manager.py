import inspect
from typing import Callable, List


from torch.fx import GraphModule
from pass_manager import PassManager


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
    ):
        if pass_managers:
            self.pass_managers = pass_managers

    def add_checks(self, check: Callable) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        sig = inspect.signature(check)

        params = [p for p in sig.parameters.values()]
        if len(params) != 1:
            raise TypeError("PassManager check function should only take in one variable, a graph_module")

        if sig.return_annotation: 
            raise TypeError("PassManager check function should return None")

        self.assert_invariants = check

    def assert_invariants(self, graph_module: GraphModule) -> None:
        pass

    def __call__(self, graph_module: GraphModule) -> None:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module to ensure that it still maintains the same required
        invariants.
        """
        
        self.assert_invariants(graph_module)

        for pm in self.pass_managers:
            pm(graph_module)
            
            if self.run_checks_after_each_pass:
                self.assert_invariants(graph_module)

        graph_module.recompile()
