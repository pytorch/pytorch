# mypy: allow-untyped-defs
import os
from typing import Callable, Optional, TypeVar

from torch._logging._internal import trace_structured_artifact
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
        *,
        context: int = 1,
    ):
        """
        log_url is inferred to be torch._inductor.config.trace.log_url_for_graph_xform unless otherwise specified
        """

        self.gm = gm
        self.passname = passname
        self.subsystem = subsystem

        if log_url is None:
            from torch._inductor.config import trace

            log_url = trace.log_url_for_graph_xform

        self.log_url = log_url
        self.context = context
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

        # initial state of the graph
        self.initial_nodes = {node.name: node for node in self.gm.graph.nodes}
        return self

    def __exit__(self, type, value, tb):
        if self.log_url is None or self.gm is None:
            return

        initial_node_names = set(self.initial_nodes.keys())

        # Capture the final state of the graph
        final_node_names = set(node.name for node in self.gm.graph.nodes)
        final_nodes = {node.name: node for node in self.gm.graph.nodes}

        added_nodes = final_node_names - initial_node_names
        removed_nodes = initial_node_names - final_node_names

        if not (added_nodes or removed_nodes):
            return

        delta_node_names = added_nodes.union(removed_nodes)
        context_node_depths = {}

        def bfs(start_node_names, max_depth):
            visited = {}
            queue = [(name, 0) for name in start_node_names]
            while queue:
                current_name, depth = queue.pop(0)
                if depth > max_depth or current_name in visited:
                    continue
                visited[current_name] = depth
                if depth > 0:
                    context_node_depths[current_name] = depth

                current_node = final_nodes.get(current_name) or self.initial_nodes.get(current_name)
                if current_node is None:
                    continue

                for arg in current_node.all_input_nodes:
                    queue.append((arg.name, depth + 1))
                for user in current_node.users:
                    queue.append((user.name, depth + 1))

        bfs(delta_node_names, self.context)

        delta_graph = FxGraphDrawer(
            self.gm,
            self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()

        nodes_to_remove = []
        for e in delta_graph.get_node_list():
            node_name = e.get_name()
            if node_name in added_nodes:
                e.obj_dict["attributes"]["fillcolor"] = "green"
            elif node_name in removed_nodes:
                e.obj_dict["attributes"]["fillcolor"] = "red"
            elif node_name in context_node_depths:
                depth = context_node_depths[node_name]
                grey_value = 100 - min(depth * 20, 80)
                e.obj_dict["attributes"]["fillcolor"] = f"grey{grey_value}"
            else:
                nodes_to_remove.append(node_name)

        # Remove the nodes not in the delta graph or context
        for node_name in nodes_to_remove:
            delta_graph.remove_node(node_name)

        output_filename = os.path.join(
            self.log_url,
            f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_delta_graph.svg",
        )
        delta_graph.write(output_filename, format="svg")

        # Log the SVG directory and filename
        svg_directory = self.log_url
        svg_filenames = [os.path.basename(output_filename)]
        trace_structured_artifact(
            name="graph_svgs",
            encoding="directory_and_filenames",
            payload_fn=lambda: {
                "directory": svg_directory,
                "filenames": svg_filenames,
            },
        )

