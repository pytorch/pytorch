from .graph_module import GraphModule
from .graph import Graph
from .node import Argument, Node, map_arg
from .proxy import Proxy
from .symbolic_trace import Tracer
from typing import Any, Dict, Iterator, Tuple

class Interpreter:
    """
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overrideable methods
    in terms of call hierarchy:

    run()
        +-- run_node
            +-- placeholder()
            +-- get_attr()
            +-- call_function()
            +-- call_method()
            +-- call_module
            +-- output()

    Example:
        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(self, n : Node) -> Any:
                    if n.target == torch.sigmoid:
                        n = copy.copy(n)
                        n.target = torch.neg
                    return super().call_function(n)

                def call_method(self, n : Node) -> Any:
                    if n.target == 'neg':
                        n = copy.copy(n)
                        n.target = 'sigmoid'
                    return super().call_method(n)

            def fn(x):
                return torch.sigmoid(x).neg()

            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            assert torch.testing.assert_allclose(result, torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The module to be executed
    """
    def __init__(self, module : GraphModule):
        assert isinstance(module, GraphModule)
        self.module = module
        self.submodules = dict(self.module.named_modules())
        self.env : Dict[Node, Any] = {}

        self.clear_env_on_return : bool = True

    def run(self, *args) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order

        Returns:
            Any: The value returned from executing the Module
        """
        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        self.args_iter : Iterator[Any] = iter(args)

        for node in self.module.graph.nodes:
            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            self.env[node] = self.run_node(node)

            if node.op == 'output':
                output_val = self.env[node]
                if self.clear_env_on_return:
                    self.env = {}
                return output_val

    def run_node(self, n : Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        return getattr(self, n.op)(n)

    # Main Node running APIs

    def placeholder(self, n : Node) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            n (Node): The placeholder node to run

        Returns:
            Any: The argument value that was retrieved.
        """
        return next(self.args_iter)

    def get_attr(self, n : Node) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            n (Node): The get_attr node to run

        Return:
            Any: The value of the attribute that was retrieved
        """
        assert isinstance(n.target, str)
        return self.fetch_attr(n.target)

    def call_function(self, n : Node) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            n (Node): The call_function node to run

        Return
            Any: The value returned by the function invocation
        """
        # Retrieve executed args and kwargs values from the environment
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        assert not isinstance(n.target, str)

        # Execute the function and return the result
        return n.target(*args, **kwargs)

    def call_method(self, n : Node) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            n (Node): The call_method node to run

        Return
            Any: The value returned by the method invocation
        """
        # Retrieve executed args and kwargs values from the environment
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        assert isinstance(n.target, str)
        return getattr(self_obj, n.target)(*args_tail, **kwargs)

    def call_module(self, n : Node) -> Any:
        """
        Execute a ``call_module`` node and return the result.

        Args:
            n (Node): The call_module node to run

        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        # Execute the method and return the result
        assert isinstance(kwargs, dict)
        if n.target not in self.submodules:
            raise RuntimeError(f'Node {n} referenced nonexistent submodule {n.target}!')

        return self.submodules[n.target](*args, **kwargs)

    def output(self, n : Node) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            n (Node): The output node to run

        Return:
            Any: The return value referenced by the output node
        """
        args, _ = self.fetch_args_kwargs_from_env(n)
        return args[0]

    # Helper methods

    def fetch_attr(self, target : str):
        """
        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualfiied name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
        target_atoms = target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def fetch_args_kwargs_from_env(self, n : Node) -> Tuple[Tuple, Dict]:
        """
        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
        args = self.map_nodes_to_values(n.args, n)
        assert isinstance(args, tuple)
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        assert isinstance(kwargs, dict)
        return args, kwargs

    def map_nodes_to_values(self, args : Argument, n : Node) -> Argument:
        """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """
        def load_arg(n_arg : Node) -> Any:
            if n_arg not in self.env:
                raise RuntimeError(f'Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() '
                                   f'to diagnose such issues')
            return self.env[n_arg]
        return map_arg(args, load_arg)

class TransformerTracer(Tracer):
    def __init__(self, graph: Graph):
        super().__init__()
        self.graph = graph

    def is_leaf_module(self, _, __) -> bool:
        return True

class Transformer(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.new_graph = Graph()
        self.tracer = TransformerTracer(self.new_graph)
        self.tracer.root = module

    def placeholder(self, n : Node) -> Proxy:
        assert isinstance(n.target, str)
        return Proxy(self.new_graph.placeholder(n.target), self.tracer)

    def get_attr(self, n : Node) -> Proxy:
        assert isinstance(n.target, str)
        return Proxy(self.new_graph.get_attr(n.target), self.tracer)

    def transform(self) -> GraphModule:
        result = super().run()
        if result is not None:
            assert isinstance(result, Proxy)
            self.new_graph.output(result.node)
        return GraphModule(self.module, self.new_graph)
