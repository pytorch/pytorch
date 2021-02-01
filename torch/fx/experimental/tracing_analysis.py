from typing import Any, Dict, Iterable, List, Callable

import torch
from torch.fx.node import Node
from torch.fx.symbolic_trace import GraphModule


class TracingAnalysis(object):
    """
    `TracingAnalysis` is designed to be subclassed in order to build
    FX analysis passes that operate by running an example input through
    the graph and observing values along the way.

    As an example, you could implement shape propagation via:

        class ShapeProp(TracingAnalysis):
            def store_result(self, node, result):
                if isinstance(result, torch.Tensor):
                    node.shape = result.shape
                    node.dtype = result.dtype
                super().store_result(node, result)

        mod = torch.fx.symbolic_trace(...)
        ShapeProp(mod).run(*example_input)
        # nodes in mod now contain `.shape` and `.dtype`
    """

    def __init__(self, module: GraphModule):
        """
        Construct a `TracingAnalysis` object.

        Args:

            module (torch.fx.GraphModule): module to run analysis on.
        """
        super(TracingAnalysis, self).__init__()
        self.module: GraphModule = module
        self.env: Dict[str, Any] = dict()
        self._named_modules: Dict[str, torch.Module] = dict()
        self._args_iter: Iterable = iter([])

    def run(self, *args: List[Any]) -> Any:
        """
        Run this analysis and execute self.module.

        Args:
            *args: Args to pass to self.module

        Returns:
            The return value of self.module
        """
        self._args_iter = iter(args)
        self._named_modules: Dict[str, torch.Module] = dict(self.module.named_modules())
        try:
            for node in self.module.graph.nodes:
                self.before_node(node)
                result = getattr(self, f"run_{node.op}")(node)
                self.store_result(node, result)
                if node.op == 'output':
                    return result
        finally:
            self._args_iter = iter([])
            self._named_modules.clear()
            self.env.clear()

    def run_placeholder(self, node: Node) -> Any:
        """
        Execute a "placeholder" node in the graph.

        Args:
            node (torch.fx.node.Node): node the in graph current being run

        Returns:
            The next arg passed to the graph
        """
        return next(self._args_iter)

    def run_get_attr(self, node: Node) -> Any:
        """
        Execute a "get_attr" node in the graph.

        Args:
            node (torch.fx.node.Node): node the in graph current being run

        Returns:
            The value of the loaded attribute
        """
        target = node.target
        target_atoms = target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced non-existant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def run_call_function(self, node: Node) -> Any:
        """
        Execute a "call_function" node in the graph.

        Args:
            node (torch.fx.node.Node): node the in graph current being run

        Returns:
            The return value of the called function
        """
        return self.run_call_any(node,
                                 node.target,
                                 self.load_args(node, node.args),
                                 self.load_args(node, node.kwargs))

    def run_call_method(self, node: Node) -> Any:
        """
        Execute a "call_method" node in the graph.

        Args:
            node (torch.fx.node.Node): node the in graph current being run

        Returns:
            The return value of the called method
        """
        self_obj, *args = self.load_args(node, node.args)
        kwargs = self.load_args(node, node.kwargs)
        return self.run_call_any(node,
                                 getattr(self_obj, node.target),
                                 args,
                                 kwargs)

    def run_call_module(self, node: Node) -> Any:
        """
        Execute a "call_module" node in the graph.

        Args:
            node (torch.fx.node.Node): node the in graph current being run

        Returns:
            The return value of the called module
        """
        return self.run_call_any(node,
                                 self._named_modules[node.target],
                                 self.load_args(node, node.args),
                                 self.load_args(node, node.kwargs))

    def run_call_any(self, node: Node, fn: Callable, args: List[Any], kwargs: Dict[str, Any]) -> Any:
        """
        Common hook used by "call_function", "call_method", and
        "call_module" nodes in graph.  This is meant to be overridden
        to add analysis around all call_* nodes.

        Args:
            node (torch.fx.node.Node): node the in graph current being run
            fn (Callable): function, method, or module to call
            args (List[Any]): *args to pass to callable
            kwargs (Dict[str, Any]): **kwargs to pass to callable

        Returns:
            The return value of fn(*args, **kwargs)
        """
        return fn(*args, **kwargs)

    def run_output(self, node: Node) -> Any:
        """
        Execute a "output" node in the graph.

        Args:
            node (torch.fx.node.Node): node the in graph current being run

        Returns:
            The return value of the graph
        """
        return self.load_args(node, node.args)[0]

    def load_args(self, node: Node, arg: Any) -> Any:
        """
        Load the args for node. `arg` will be either node.args
        or node.kwargs and may be a nested data structure of
        dict/list/tuple/slice.  Leaf nodes in that data structure are
        pointers to other nodes in the graph.

        Args:
            node (torch.fx.node.Node): node the in graph current being run
            arg (Any): args that should be loaded

        Returns:
            Loaded args
        """
        return torch.fx.node.map_arg(arg, self.load)

    def load(self, node: Node):
        """
        Load the output value of a node that has previously been executed.

        Args:
            node (torch.fx.node.Node): node to read the output of

        Returns:
            the output value of node from `self.env`
        """
        return self.env[node.name]

    def store_result(self, node: Node, result: Any):
        """
        This is run after executing each node and stores the output of
        that node in `self.env`

        Args:
            node (torch.fx.node.Node): node the in graph current being run
            result (Any): the output value generated by node
        """
        self.env[node.name] = result

    def before_node(self, node: Node):
        """
        Hook to allow custom code to be run before executing a node.

        Args:
            node (torch.fx.node.Node): node the in graph current being run
        """
        pass
