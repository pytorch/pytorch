import copy
import itertools
import logging
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Type

import torch.nn as nn
from torch import fx
from torch.fx.graph import PythonCode
from torch.fx.node import Argument
from torch.utils._pytree import tree_flatten, tree_map


logger: logging.Logger = logging.getLogger("IterGraphModule")


class IterGraph(fx.Graph):
    """
    ``IterGraph`` is used to perform cross-iteration optimization. ``IterGraph``
    keeps track of the 3 graphs, self (the original graph), setup graph, and
    cleanup graph. The 3 graphs should be identical copies of a ``fx.Graph``.

    IterGraph subclass fx.Graph to override the necessary APIs that will be used
    when constructing a optimization, e.g., communication fusion. IterGraph also
    provides APIs that originally belong to fx.Node and all these APIs will have
    ``node_`` prefix. For example, ``IterGraph.node_prepend`` is the equivalance
    of ``fx.Node.prepend``. Note that all the optimizations must be constructed
    using these APIs.
    """

    def __init__(
        self,
        orig_graph: fx.Graph,
        setup_graph: fx.Graph,
        cleanup_graph: fx.Graph,
        owning_module: Optional[fx.GraphModule] = None,
        tracer_cls: Optional[Type["fx.Tracer"]] = None,
        tracer_extras: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(owning_module, tracer_cls, tracer_extras)

        output_vals = self.graph_copy(orig_graph, {}, return_output_node=True)
        # TODO: if we do ``deepcopy(_codegen)`` and the input argument contains
        # a dictionary with the form of Dict[torch.Tensor, Any], the
        # torch.fx._pytree.treen_flatten_spec will not be able to flatten the
        # dict -- the torch.Tensor will be duplicated because the _input_spec
        # will save the ``keys`` of a dictionary (the values are not saved).
        self._codegen = copy.deepcopy(orig_graph._codegen)
        assert isinstance(output_vals, tuple)
        output_val, old_output_val = output_vals
        super().output(output_val, type_expr=getattr(old_output_val, "type", None))

        self.setup_graph = setup_graph
        self.cleanup_graph = cleanup_graph
        self._all_graphs: Tuple[fx.Graph, ...] = (
            self.setup_graph,
            self.cleanup_graph,
            cast(fx.Graph, super()),
        )

        self._setup_mapping: Dict[fx.Node, fx.Node] = {}
        self._cleanup_mapping: Dict[fx.Node, fx.Node] = {}
        self._freeze_cross_iter_movement = False

        for node, setup_node, cleanup_node in zip(
            self.nodes, self.setup_graph.nodes, self.cleanup_graph.nodes
        ):
            self._setup_mapping[node] = setup_node
            self._cleanup_mapping[node] = cleanup_node

        self.num_extra_output = 0

    def _lookup_node(self, node: fx.Node, graph: fx.Graph) -> Optional[fx.Node]:
        if graph == self.setup_graph:
            return self._setup_mapping.get(node, None)
        elif graph == self.cleanup_graph:
            return self._cleanup_mapping.get(node, None)
        return node

    def _insert_context(self, func: str, node: fx.Node):
        with ExitStack() as stack:
            for graph in self._all_graphs:
                if node:
                    actual_node = self._lookup_node(node, graph)
                    assert actual_node is not None, "Cannot handle None case now."
                else:
                    actual_node = node
                stack.enter_context(getattr(graph, func)(actual_node))
            yield

    def _fx_graph_call(
        self, graph: fx.Graph, func: str, *args: Any, **kwargs: Any
    ) -> Any:
        fx_graph: fx.Graph = graph if graph != self else cast(fx.Graph, super())
        return getattr(fx_graph, func)(*args, **kwargs)

    @contextmanager
    def inserting_after(self, node):
        return self._insert_context("inserting_after", node)

    @contextmanager
    def inserting_before(self, node):
        return self._insert_context("inserting_before", node)

    @staticmethod
    def _is_sese_graph(subgraph: List[fx.Node]) -> bool:
        """
        Check if the given subgraph forms a single-entry-single-exit (SESE) graph.
        We borrow the SESE graph term from the graph theory because the graph
        looks similar to SESE graph.

        1. Only one node has inputs/args from the external nodes (not in subgraph).
        2. Only one node has a user from the the external nodes and the user must
           be output. The output condition is not strictly enforced due to the
           testing purpose.

        SESE graph is very restrict and is not applicable for all optimizations.
        But this gives a good start point to explor cross-iteration optimizations.
        """
        all_nodes: Set[fx.Node] = set(subgraph)
        for i, node in enumerate(subgraph):
            pytree_args, _ = tree_flatten(node.args)
            pytree_kwargs, _ = tree_flatten(node.kwargs)
            for arg in itertools.chain(pytree_args, pytree_kwargs):
                if arg not in all_nodes and i > 0:
                    return False
            if i == len(subgraph) - 1:
                # TODO: the only user must be the output. Otherwise, we don't
                # know how to move this subgraph. We currently do not stricly
                # force this attribute because some test code create fake output.
                if len(node.users) > 1:
                    return False
            else:
                for user in node.users:
                    if user not in all_nodes:
                        return False
        return True

    def _convert_sese_input_to_output(
        self, nodes: List[fx.Node], graph: fx.Graph, erase_node: bool
    ) -> None:
        for output in reversed(graph.nodes):
            if output.target == "output":
                break
        first_node = self._lookup_node(nodes[0], graph)
        assert first_node is not None, "The first_node is None."
        # TODO: We currently assume there is only one input to simplify the
        # coding. We may have to deal with a more general case.
        sese_arguments = first_node.args[0]
        # TODO: Do we need to remove the output of the SESE subgraph?
        # new_output =  tuple(
        #   arg for arg in output.args if arg != nodes[-1]
        # ) + (arguments,)
        new_output = output.args + (sese_arguments,)
        if erase_node:
            for node in nodes:
                graph_node = self._lookup_node(node, graph)
                assert graph_node is not None, "The graph_node is None."
                self._fx_graph_call(graph, "erase_node", graph_node)
        self._fx_graph_call(graph, "erase_node", output)
        self._fx_graph_call(graph, "output", new_output)

    def move_to_next_iter_before(self, nodes: List[fx.Node], target_node: fx.Node):
        if self._freeze_cross_iter_movement:
            raise RuntimeError(
                "The cross-iteration movement has been freeze for the given "
                "IterGraph."
            )

        if not self._is_sese_graph(nodes):
            raise ValueError(
                "The nodes for move_to_next_iter_before must form a SESE "
                "subgraph. The output of this subgraph should be the output "
                " of the whole graph."
            )

        # The main graph must be the last one to be modified. Otherwise, the
        # mapping may change and hence intorduce incorrect mapping for setup
        # and cleanup graphs.

        # For the setup graph, no additional input is needed but additional
        # outputs will be created. The additional output represents the input of
        # the action to be moved to the next iteration -- main graph.
        self._convert_sese_input_to_output(
            nodes=nodes, graph=self.setup_graph, erase_node=True
        )

        # For the cleanup graph, additional input is required to get the output
        # from the last iteration -- main graph. Additional nodes are also
        # needed to perform the action moved from the last itertion.
        new_input_node = self.cleanup_graph.placeholder(nodes[0].name + "_input")
        target_cleanup_node = self._lookup_node(target_node, self.cleanup_graph)
        assert target_cleanup_node is not None, "The target_cleanup_node is None."
        node_mapping: Dict[fx.Node, fx.Node] = {}
        with self.cleanup_graph.inserting_before(target_cleanup_node):
            last_new_cleanup_node: Optional[fx.Node] = None
            for i, node in enumerate(nodes):
                cleanup_node = self._lookup_node(node, self.cleanup_graph)
                assert cleanup_node is not None, "The cleanup_node is None."
                # TODO: generalize the node copy process. We only support
                # call_function now and trivial args, kwargs for the first node.
                if i == 0:
                    args = (new_input_node,)
                    kwargs = {}
                else:
                    args = tree_map(
                        lambda arg: node_mapping[arg]
                        if isinstance(arg, fx.Node)
                        else arg,
                        cleanup_node.args,
                    )
                    kwargs = tree_map(
                        lambda arg: node_mapping[arg]
                        if isinstance(arg, fx.Node)
                        else arg,
                        cleanup_node.kwargs,
                    )
                new_cleanup_node = self.cleanup_graph.call_function(
                    cleanup_node.target,
                    args,
                    kwargs,
                    cleanup_node.type,
                )
                if i == 0:
                    new_cleanup_node.prepend(new_input_node)
                node_mapping[cleanup_node] = new_cleanup_node
                last_new_cleanup_node = new_cleanup_node
            assert last_new_cleanup_node is not None
            # TODO: Figure out how to properly avoid dead code elimination that
            # clean up the newly added node properly. Right now, we manually
            # update the users of the last node of the new SESE graph even
            # though the target node does not use that node.
            last_new_cleanup_node.users[target_cleanup_node] = None

        # For the main graph, additional input will be created to represent
        # the output from the last iteration -- main graph or setup graph.
        # Additional output will also be generated to represent the input for
        # the next iteration -- the main graph or the cleanup graph.
        self._convert_sese_input_to_output(nodes=nodes, graph=self, erase_node=False)
        new_input_node = self.placeholder(nodes[0].name + "_input")
        nodes[0].args = (new_input_node,)
        for node in nodes:
            target_node.prepend(node)
        nodes[0].prepend(new_input_node)
        nodes[-1].users[target_node] = None

        self.num_extra_output += 1

    def move_before(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        for graph in self._all_graphs:
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None
            for actual_node in actual_nodes:
                actual_target_node.prepend(actual_node)

    def move_after(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        for graph in self._all_graphs:
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            actual_target_node = self._lookup_node(target_node, graph)
            for actual_node in actual_nodes:
                assert actual_target_node is not None
                actual_target_node.append(actual_node)
                actual_target_node = actual_node

    def call_function(
        self,
        the_function: Callable[..., Any],
        args: Optional[Tuple[Argument, ...]] = None,
        kwargs: Optional[Dict[str, Argument]] = None,
        type_expr: Optional[Any] = None,
    ) -> fx.Node:
        if self._freeze_cross_iter_movement:
            return super().call_function(the_function, args, kwargs, type_expr)

        setup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        setup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )
        cleanup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        cleanup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )

        setup_node = self.setup_graph.call_function(
            the_function, setup_args, setup_kwargs, type_expr
        )
        main_node = super().call_function(the_function, args, kwargs, type_expr)
        cleanup_node = self.cleanup_graph.call_function(
            the_function, cleanup_args, cleanup_kwargs, type_expr
        )
        self._setup_mapping[main_node] = setup_node
        self._cleanup_mapping[main_node] = cleanup_node
        return main_node

    def erase_node(self, to_erase: fx.Node) -> None:
        if self._freeze_cross_iter_movement:
            return super().erase_node(to_erase)

        setup_node = self._lookup_node(to_erase, self.setup_graph)
        assert setup_node is not None, "setup_node is None"
        self.setup_graph.erase_node(setup_node)
        super().erase_node(to_erase)
        cleanup_node = self._lookup_node(to_erase, self.cleanup_graph)
        self.cleanup_graph.erase_node(cleanup_node)

    def output(self, result: Argument, type_expr: Optional[Any] = None) -> fx.Node:
        if self._freeze_cross_iter_movement:
            return super().output(result, type_expr)

        main_output = super().output(result, type_expr)
        setup_result = tree_map(
            lambda _result: self._lookup_node(_result, self.setup_graph)
            if isinstance(_result, fx.Node)
            else _result,
            result,
        )
        cleanup_result = tree_map(
            lambda _result: self._lookup_node(_result, self.cleanup_graph)
            if isinstance(_result, fx.Node)
            else _result,
            result,
        )
        self.setup_graph.output(setup_result, type_expr)
        self.cleanup_graph.output(cleanup_result, type_expr)

        return main_output

    def lint(self) -> None:
        self.setup_graph.lint()
        super().lint()
        self.cleanup_graph.lint()

    def node_prepend(self, target_node: fx.Node, node: fx.Node) -> None:
        """Prepend node to target_node."""
        if self._freeze_cross_iter_movement:
            target_node.prepend(node)
            return

        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, "The node is None"
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None, "The target node is None"
            actual_target_node.prepend(actual_node)

    def node_append(self, target_node: fx.Node, node: fx.Node) -> None:
        """Append node to target_node."""
        if self._freeze_cross_iter_movement:
            target_node.append(node)
            return

        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, f"The actual node is None, {node}."
            actual_target_node = self._lookup_node(target_node, graph)
            assert (
                actual_target_node is not None
            ), f"The actual target node is None, {target_node}."
            actual_target_node.append(actual_node)

    def node_update_arg(self, node: fx.Node, idx: int, arg: Argument) -> None:
        if self._freeze_cross_iter_movement:
            node.update_arg(int, arg)
            return

        setup_arg = tree_map(
            lambda _arg: self._lookup_node(_arg, self.setup_graph)
            if isinstance(_arg, fx.Node)
            else _arg,
            arg,
        )
        setup_node = self._lookup_node(node, self.setup_graph)
        assert setup_node is not None, "setup_node is None"
        setup_node.update_arg(idx, setup_arg)

        node.update_arg(idx, arg)

        cleanup_arg = tree_map(
            lambda _arg: self._lookup_node(_arg, self.cleanup_graph)
            if isinstance(_arg, fx.Node)
            else _arg,
            arg,
        )
        cleanup_node = self._lookup_node(node, self.cleanup_graph)
        assert cleanup_node is not None, "cleanup_node is None"
        cleanup_node.update_arg(idx, cleanup_arg)

    def node_replace_all_uses_with(
        self,
        node: fx.Node,
        replace_with: fx.Node,
        delete_user_cb: Callable[[fx.Node], bool] = lambda user: True,
        *,
        propagate_meta=False,
    ) -> List[fx.Node]:
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            actual_replace_with = self._lookup_node(replace_with, graph)
            assert actual_node is not None
            ret = actual_node.replace_all_uses_with(
                actual_replace_with, delete_user_cb, propagate_meta=propagate_meta
            )
        return ret

    def node_add_user(self, node: fx.Node, user: Any) -> None:
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            assert actual_node is not None
            actual_node.users[actual_user_node] = None  # type: ignore[index]

    def node_remove_user(self, node: fx.Node, user: Any) -> None:
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            assert actual_node is not None
            del actual_node.users[actual_user_node]  # type: ignore[arg-type]

    def keep_unused_nodes(self) -> None:
        for node in self.nodes:
            if len(node.users) == 0:
                self.node_add_user(node, "__hold__")

    def functionalize_optim(self) -> None:
        # IterGraph can only support full graph (fwd+bwd+optim). As optimizer
        # is not a functional call (it is inplace op), this mehod adds the of
        # the optimizer call. This method has strong assumption of the optimizer
        # and may not always be working. This method is intended be a temporary
        # solution only.
        for node in reversed(self.nodes):
            if node.name.startswith("output"):
                output_node = node
            elif node.name.startswith("_fused_adam_",):
                optim_node = node
            elif node.name.startswith("_foreach_add_",):
                step_node = node
                self.node_add_user(optim_node, output_node)
                self.node_add_user(step_node, optim_node)

    def defunctionalize_optim(self) -> None:
        for i, node in enumerate(reversed(self.nodes)):
            if node.name.startswith("output"):
                output_node = node
            elif node.name.startswith("_fused_adam_",):
                optim_node = node
            elif node.name.startswith("_foreach_add_",):
                step_node = node
                self.node_add_user(step_node, optim_node)
                self.node_remove_user(optim_node, output_node)
                self.node_remove_user(step_node, optim_node)

    def freeze_cross_iter_movement(self) -> None:
        self._freeze_cross_iter_movement = True


class IterGraphModule(nn.Module):
    """
    ``IterGraphModule`` provides the ability to do cross-iteration optimization.
    Given a ``fx.GraphModule``, main_gm, ``IterGraphModule`` internally
    duplicate it to 3 copies and redirect the ``forward`` request to a different
    ``fx.GraphModule`` based on the iteration count. This allows users to do
    graph optimizations that across iterations (e.g., moving collective wait in
    the backward to the forward of the next iteration).

    Note that users must call the APIs provided by ``IterGraphModule`` or
    ``IterGraph`` to rewrite the graph so that ``IterGraphModule`` can keep the
    data dependency for all 3 graphs.
    """

    def __init__(self, main_gm: fx.GraphModule) -> None:
        super().__init__()

        def _copy_gm(src: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
            gm = fx.GraphModule(src, graph)
            gm.meta = getattr(graph, "meta", {})
            return gm

        self.setup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.cleanup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.main_gm = _copy_gm(
            main_gm,
            IterGraph(main_gm.graph, self.setup_gm.graph, self.cleanup_gm.graph),
        )

        self._iter = 0
        self._max_iters = 0
        self._previous_output: Tuple[Any, ...] = tuple()

    def setup(self, max_iters: int = 0) -> None:
        """
        This method is used to tell IterGraphModule the iterations to train so
        that IterGraphModule knows which iteration is the last one and can do
        proper cleanup.
        """
        # TODO: There are cases where max_iters is not known or not precise,
        # e.g., data is depleted. One suggestion from the reviewer is to
        # add one extra argument to forward(..., last_iter: bool = False) to
        # allow users to tell us if the last iteration happens.
        if max_iters <= 0:
            raise ValueError(f"Incorrect max_iters is set, {max_iters}")
        self._iter = 0
        self._max_iters = max_iters

    def _run(self, gm: fx.GraphModule, *args, **kwargs) -> Any:
        if cast(IterGraph, self.main_gm.graph).num_extra_output > 0:
            # TODO: a general way to support different types of input and output.
            assert not kwargs, "Has not supported kwargs now."
            new_args = args + (self._previous_output)
            output = gm(*new_args, **kwargs)
            if self._iter < self._max_iters:
                assert isinstance(
                    output, tuple
                ), f"Only support tuple output now. {type(output)}"
                num_actual_output = (
                    len(output) - cast(IterGraph, self.main_gm.graph).num_extra_output
                )
                assert num_actual_output > 0
                self._previous_output = output[num_actual_output:]
                output = output[:num_actual_output]
                if len(output) == 1:
                    output = output[0]
        else:
            # No cross-iteration optimization is done. Simply call the
            # GraphModule.
            output = gm(*args, **kwargs)
        logger.info(f"The output information: size={len(output)}, type={type(output)}")
        return output

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._iter += 1
        if self._iter == 1:
            self.print_all_graphs()
            logger.warning("Using the setup graph")
            gm = self.setup_gm
        elif self._iter == self._max_iters:
            logger.warning("Using the cleanup graph")
            gm = self.cleanup_gm
        else:
            logger.warning("Using the main graph")
            gm = self.main_gm

        return self._run(gm, *args, **kwargs)

    @property
    def graph(self) -> IterGraph:
        return cast(IterGraph, self.main_gm.graph)

    def recompile(self) -> PythonCode:
        self.setup_gm.recompile()
        self.cleanup_gm.recompile()
        return self.main_gm.recompile()

    def print_readable(self, print_output: bool = True) -> str:
        return self.main_gm.print_readable(print_output)

    def print_all_graphs(self) -> None:
        logger.info("Printing the three fx.Graph:")
        logger.info("1. Setup fx.Graph:")
        logger.info(f"{self.setup_gm.graph}")
        logger.info("2. Main fx.Graph:")
        logger.info(f"{self.main_gm.graph}")
        logger.info("3. Cleanup fx.Graph:")
        logger.info(f"{self.cleanup_gm.graph}")

    def print_all_graph_modules(self) -> None:
        logger.info("Printing the three fx gm:")
        logger.info("1. Setup fx.GraphModule:")
        logger.info(f"{self.setup_gm.print_readable(False)}")
        logger.info("2. Main fx.GraphModule:")
        logger.info(f"{self.main_gm.print_readable(False)}")
        logger.info("3. Cleanup fx.GraphModule:")
        logger.info(f"{self.cleanup_gm.print_readable(False)}")
