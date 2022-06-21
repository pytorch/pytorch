import abc

import torch.fx as fx


class PassBase(abc.ABC):
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, graph_module: fx.GraphModule) -> None:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """

        self.requires(graph_module)
        self.call(graph_module)
        self.ensures(graph_module)

    @abc.abstractmethod
    def call(self, graph_module: fx.GraphModule) -> None:
        """
        The pass that is run through the given graph module. To implement a
        pass, it is required to implement this function.

        Args:
            graph_module: The graph module we will run a pass on
        """
        pass

    def requires(self, graph_module: fx.GraphModule) -> None:
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
        pass

    def ensures(self, graph_module: fx.GraphModule) -> None:
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
        pass
