# mypy: allow-untyped-defs
import os
from typing import Callable, Optional, TypeVar

from torch.fx import Graph
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule


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
        if self.log_url is None or self.gm is None:
            return self

        self.erased_nodes = set()
        self.created_nodes = set()
        self.gm._register_create_node_hook(self.on_node_creation)
        self.gm._register_erase_node_hook(self.on_node_erase)

        return self

    def __exit__(self, type, value, tb):
        if self.log_url is None or self.gm is None:
            return

        self.gm._unregister_create_node_hook(self.on_node_creation)
        self.gm._unregister_erase_node_hook(self.on_node_erase)

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

    def on_node_erase(self, node):
        self.erased_nodes.add(node.name)
