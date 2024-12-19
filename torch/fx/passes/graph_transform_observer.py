# mypy: allow-untyped-defs
import os
from typing import Callable, Optional, TypeVar

from torch.fx import Graph, Node
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.traceback import NodeSource, NodeSourceAction


T = TypeVar("T")


from .graph_drawer import FxGraphDrawer


__all__ = ["GraphTransformObserver"]


@compatibility(is_backward_compatible=False)
class GraphTransformObserver:
    __pass_count = 0

    def __init__(
        self,
        gm: GraphModule,
        passname: str,
        subsystem: Optional[str] = None,
        log_url: Optional[str] = None,
    ):
        """
        log_url is inferred to be torch._inductor.config.trace.log_url_for_graph_xform unless otherwise specified
        """

        self.gm = gm
        self.passname = passname
        self.subsystem = subsystem

        self.erased_nodes = set()
        self.created_nodes = set()
        self.name_to_node = {}

        # If log_url is None, we don't log anything
        if log_url is None:
            from torch._inductor.config import trace

            log_url = trace.log_url_for_graph_xform

        self.log_url = log_url
        if self.log_url is None:
            return
        GraphTransformObserver.__pass_count += 1

        self.input_dot_graph = FxGraphDrawer(
            self.gm,
            self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()

    @classmethod
    def get_current_pass_count(cls):
        return cls.__pass_count

    def apply_gm_pass(self, pass_fn: Callable[[GraphModule], T]) -> Optional[T]:
        with self:
            if not self._check_disable_pass():
                return pass_fn(self.gm)

        return None

    def apply_graph_pass(self, pass_fn: Callable[[Graph], T]) -> Optional[T]:
        with self:
            if not self._check_disable_pass():
                return pass_fn(self.gm.graph)

        return None

    def _check_disable_pass(self):
        if self.subsystem is None:
            return False

        debug_info = lambda: self.passname  # noqa: E731
        from torch._inductor.compiler_bisector import CompilerBisector

        return CompilerBisector.disable_subsystem(
            "inductor", self.subsystem, debug_info
        )

    def __enter__(self):
        self.gm._register_create_node_hook(self.on_node_creation)
        self.gm._register_erase_node_hook(self.on_node_erase)
        self.gm._register_replace_node_hook(self.on_node_replace)

        self.erased_nodes.clear()
        self.created_nodes.clear()
        self.name_to_node.clear()

        for node in self.gm.graph.nodes:
            self.name_to_node[node.name] = node

        return self

    def __exit__(self, type, value, tb):
        self.gm._unregister_create_node_hook(self.on_node_creation)
        self.gm._unregister_erase_node_hook(self.on_node_erase)
        self.gm._unregister_replace_node_hook(self.on_node_replace)

        if self.log_url is None or self.gm is None:
            return

        if len(self.created_nodes) > 0 or len(self.erased_nodes) > 0:
            for e in self.input_dot_graph.get_node_list():
                if e.get_name() in self.erased_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            self.input_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_input_graph.dot",
                )
            )

            output_dot_graph = FxGraphDrawer(
                self.gm,
                self.passname,
                ignore_getattr=True,
                ignore_parameters_and_buffers=True,
            ).get_dot_graph()
            for e in output_dot_graph.get_node_list():
                if e.get_name() in self.created_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            output_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_output_graph.dot",
                )
            )

    def on_node_creation(self, node):
        self.created_nodes.add(node.name)
        self.name_to_node[node.name] = node
        source = NodeSource(None, self.passname, NodeSourceAction.CREATE)
        if "from_node" not in node.meta:
            node.meta["from_node"] = [source]
        else:
            node.meta["from_node"].append(source)

    def on_node_erase(self, node):
        self.erased_nodes.add(node.name)
        self.name_to_node.pop(node.name, None)

    def on_node_replace(self, old: Node, new: str, user: Node):
        # Update node meta when replacing old node with new node
        new_node = self.name_to_node.get(new, None)

        action = [NodeSourceAction.REPLACE]
        if new_node.name in self.created_nodes:
            action.append(NodeSourceAction.CREATE)

        def created_this_pass(source):
            return source.passname == self.passname and source.action == [
                NodeSourceAction.CREATE
            ]

        # remove redundant source added on node creation
        new_from_node = new_node.meta.get("from_node", [])
        new_from_node = [
            source for source in new_from_node if not created_this_pass(source)
        ]

        # add new source
        new_node_source = NodeSource(old, self.passname, action)
        new_from_node.append(new_node_source)
        new_node.meta["from_node"] = new_from_node
