from .graph_module import GraphModule
from .graph import Graph
from .node import Argument, Node, map_arg
from .proxy import Proxy
from .symbolic_trace import Tracer
from typing import Any, Dict, Iterator, Tuple

class Interpreter:
    def __init__(self, module : GraphModule):
        assert isinstance(module, GraphModule)
        self.module = module
        self.submodules = dict(self.module.named_modules())
        self.env : Dict[Node, Any] = {}

        self.clear_env_on_return : bool = True

    def run(self, *args) -> Any:
        """
        Run `module` via interpretation and return the result.
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
        return getattr(self, n.op)(n)

    # Main Node running APIs

    def placeholder(self, n : Node) -> Any:
        return next(self.args_iter)

    def get_attr(self, n : Node) -> Any:
        assert isinstance(n.target, str)
        return self.fetch_attr(n.target)

    def call_function(self, n : Node) -> Any:
        # Retrieve executed args and kwargs values from the environment
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        # Execute the function and return the result
        return n.target(*args, **kwargs)

    def call_method(self, n : Node) -> Any:
        # Retrieve executed args and kwargs values from the environment
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        # args[0] is the `self` object for this method call
        self_obj, *args = args

        # Execute the method and return the result
        assert isinstance(n.target, str)
        return getattr(self_obj, n.target)(*args, **kwargs)

    def call_module(self, n : Node) -> Any:
        # Retrieve executed args and kwargs values from the environment
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        # Execute the method and return the result
        assert isinstance(kwargs, dict)
        if n.target not in self.submodules:
            raise RuntimeError(f'Node {n} referenced nonexistent submodule {n.target}!')

        return self.submodules[n.target](*args, **kwargs)

    def output(self, n : Node) -> Any:
        args, _ = self.fetch_args_kwargs_from_env(n)
        return args[0]

    # Helper methods

    def fetch_attr(self, target : str):
        target_atoms = target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def fetch_args_kwargs_from_env(self, n : Node) -> Tuple[Tuple, Dict]:
        args = self.map_nodes_to_values(n.args, n)
        assert isinstance(args, tuple)
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        assert isinstance(kwargs, dict)
        return args, kwargs

    def map_nodes_to_values(self, args : Argument, n : Node) -> Argument:
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
        return Proxy(self.new_graph.placeholder(n.target), self.tracer)

    def get_attr(self, n : Node) -> Proxy:
        return Proxy(self.new_graph.get_attr(n.target), self.tracer)

    def transform(self) -> GraphModule:
        result = super().run()
        if result is not None:
            assert isinstance(result, Proxy)
            self.new_graph.output(result.node)
        return GraphModule(self.module, self.new_graph)
