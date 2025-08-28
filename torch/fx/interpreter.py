# mypy: allow-untyped-defs
import inspect
from contextlib import contextmanager
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
import torch.fx.traceback as fx_traceback
from torch._logging import trace_structured
from torch.hub import tqdm

from . import config
from ._compatibility import compatibility
from ._lazy_graph_module import _make_graph_module
from ._symbolic_trace import Tracer
from .graph import Graph
from .graph_module import GraphModule
from .node import Argument, map_aggregate, map_arg, Node, Target
from .proxy import Proxy


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = ["Interpreter", "Transformer"]


@compatibility(is_backward_compatible=True)
class Interpreter:
    """
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overrideable methods
    in terms of call hierarchy::

        run()
            +-- run_node
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(
                    self, target: Target, args: Tuple, kwargs: Dict
                ) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            torch.testing.assert_close(result, torch.neg(input).sigmoid())

    Args:
        module (torch.nn.Module): The module to be executed
        garbage_collect_values (bool): Whether to delete values after their last
            use within the Module's execution. This ensures optimal memory usage during
            execution. This can be disabled to, for example, examine all of the intermediate
            values in the execution by looking at the ``Interpreter.env`` attribute.
        graph (Optional[Graph]): If passed, the interpreter will execute this
            graph instead of `module.graph`, using the provided `module`
            argument to satisfy any requests for state.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        module: torch.nn.Module,
        garbage_collect_values: bool = True,
        graph: Optional[Graph] = None,
    ):
        self.module = module
        self.submodules = dict(self.module.named_modules())
        if graph is not None:
            self.graph = graph
        else:
            self.graph = self.module.graph  # type: ignore[assignment]
        self.env: dict[Node, Any] = {}
        self.name = "Interpreter"
        self.garbage_collect_values = garbage_collect_values
        self.extra_traceback = True

        if self.garbage_collect_values:
            # Run through reverse nodes and record the first instance of a use
            # of a given node. This represents the *last* use of the node in the
            # execution order of the program, which we will use to free unused
            # values
            node_to_last_use: dict[Node, Node] = {}
            self.user_to_last_uses: dict[Node, list[Node]] = {}

            def register_last_uses(n: Node, user: Node):
                if n not in node_to_last_use:
                    node_to_last_use[n] = user
                    self.user_to_last_uses.setdefault(user, []).append(n)

            for node in reversed(self.graph.nodes):
                for n in node._input_nodes:
                    register_last_uses(n, node)

    @compatibility(is_backward_compatible=True)
    def run(
        self,
        *args,
        initial_env: Optional[dict[Node, Any]] = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = initial_env if initial_env is not None else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        self.args_iter: Iterator[Any] = iter(args)
        pbar = tqdm(
            total=len(self.graph.nodes),
            desc=f"{self.name}: {str(list(self.graph.nodes)) if config.verbose_progress else ''}",
            initial=0,
            position=0,
            leave=True,
            disable=config.disable_progress,
            delay=0,
        )

        for node in self.graph.nodes:
            pbar.update(1)
            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                if self.extra_traceback:
                    msg = f"While executing {node.format_node()}"
                    msg = f"{e.args[0]}\n\n{msg}" if e.args else str(msg)
                    msg += f"\nOriginal traceback:\n{node.stack_trace}"
                    if (
                        isinstance(self.module, GraphModule)
                        and self.module.graph is not None
                        and isinstance(self.module.graph, torch.fx.Graph)
                    ):
                        trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "fx_interpreter_error",
                                "encoding": "string",
                            },
                            payload_fn=lambda: (
                                f"{msg}\nGraphModule: "
                                f"{self.module.print_readable(print_output=False, include_stride=True)}"  # type: ignore[operator]
                            ),
                        )

                    msg += "\nUse tlparse to see full graph. "
                    msg += "(https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)"
                    e.args = (msg,) + e.args[1:]
                    if isinstance(e, KeyError):
                        raise RuntimeError(*e.args) from e
                raise

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == "output":
                output_val = self.env[node]
                return (
                    self.graph.process_outputs(output_val)
                    if enable_io_processing
                    else output_val
                )

    @compatibility(is_backward_compatible=True)
    def boxed_run(self, args_list):
        """
        Run `module` via interpretation and return the result.  This uses the "boxed"
        calling convention, where you pass a list of arguments, which will be cleared
        by the interpreter.  This ensures that input tensors are promptly deallocated.
        """
        args_iter = iter(args_list)
        env = {}
        for n in self.graph.nodes:
            if n.op == "placeholder":
                env[n] = next(args_iter)
        args_list.clear()
        return self.run(initial_env=env)

    @contextmanager
    def _set_current_node(self, node):
        with fx_traceback.set_current_meta(
            node, f"Interpreter_{self.__class__.__name__}"
        ):
            yield

    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any:
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
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            return getattr(self, n.op)(n.target, args, kwargs)

    # Main Node running APIs
    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        assert isinstance(target, str)
        if target.startswith("*"):
            # For a starred parameter e.g. `*args`, retrieve all
            # remaining values from the args list.
            return list(self.args_iter)
        else:
            try:
                return next(self.args_iter)
            except StopIteration as si:
                if len(args) > 0:
                    return args[0]
                else:
                    raise RuntimeError(
                        f"Expected positional argument for parameter {target}, but one was not passed in!"
                    ) from si

    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
        assert isinstance(target, str)
        return self.fetch_attr(target)

    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        # Execute the function and return the result
        return target(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        assert isinstance(target, str)
        return getattr(self_obj, target)(*args_tail, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        return submod(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The return value referenced by the output node
        """
        return args[0]

    # Helper methods
    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str):
        """
        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualified name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
        target_atoms = target.split(".")
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[: i + 1])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    @compatibility(is_backward_compatible=True)
    def fetch_args_kwargs_from_env(self, n: Node) -> tuple[tuple, dict]:
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

    @compatibility(is_backward_compatible=True)
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument:
        """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """

        def load_arg(n_arg: Node) -> Any:
            if n_arg not in self.env:
                raise RuntimeError(
                    f"Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() "
                    f"to diagnose such issues"
                )
            return self.env[n_arg]

        return map_arg(args, load_arg)


@compatibility(is_backward_compatible=True)
class Transformer(Interpreter):
    """
    ``Transformer`` is a special type of interpreter that produces a
    new ``Module``. It exposes a ``transform()`` method that returns
    the transformed ``Module``. ``Transformer`` does not require
    arguments to run, as ``Interpreter`` does. ``Transformer`` works
    entirely symbolically.

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass ``Transformer`` like so::

            class NegSigmSwapXformer(Transformer):
                def call_function(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target == torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)

            transformed: torch.nn.Module = NegSigmSwapXformer(gm).transform()
            input = torch.randn(3, 4)
            torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The ``Module`` to be transformed.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, module):
        super().__init__(module)
        self.new_graph = Graph()
        self.new_graph.set_codegen(module.graph._codegen)

        class TransformerTracer(Tracer):
            def __init__(self, graph: Graph):
                super().__init__()
                self.graph = graph
                self.tensor_attrs: dict[torch.Tensor, str] = {}  # type: ignore[assignment]

            def is_leaf_module(self, _, __) -> bool:
                return True

        self.tracer = TransformerTracer(self.new_graph)
        self.tracer.root = module

    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``placeholder`` node. In ``Transformer``, this is
        overridden to insert a new ``placeholder`` into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        assert isinstance(target, str)
        default_value = next(iter(args)) if args else inspect.Signature.empty
        return Proxy(
            self.new_graph.placeholder(target, default_value=default_value), self.tracer
        )

    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``get_attr`` node. In ``Transformer``, this is
        overridden to insert a new ``get_attr`` node into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        assert isinstance(target, str)
        return self.tracer.create_proxy("get_attr", target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # Override so that the leaf module policy from `self.tracer` is respected.
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        return self.tracer.call_module(submod, submod.forward, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # Override so that functions that were wrapped are still wrapped.
        return self.tracer.create_proxy("call_function", target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def transform(self) -> GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a

            new_output_node = self.new_graph.output(map_aggregate(result, strip_proxy))
            # also preserve the metadata from the old output node, if it exists
            old_output_node = list(self.graph.nodes)[-1]
            assert old_output_node.op == "output"
            for k, v in old_output_node.meta.items():
                new_output_node.meta[k] = v

        return _make_graph_module(self.module, self.new_graph)
