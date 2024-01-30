from typing import Callable

from torch import fx
from torch.distributed._spmd.graph_optimization import (
    comm_fusion_with_concat,
    enable_graph_optimization_dump,
    remove_copy_from_optimizer,
    schedule_comm_wait,
)
from torch.distributed._spmd.graph_utils import dump_graphs_to_files
from torch.distributed._spmd.iter_graph_module import IterGraphModule


class GraphModuleTransformation:
    def __init__(
        self,
        *,
        enable_graph_optimization: bool = False,
        enable_inductor: bool = False,
        dump_graphs: bool = False,
    ) -> None:
        self.enable_graph_optimization = enable_graph_optimization
        self.enable_inductor = enable_inductor
        self.dump_graphs = dump_graphs

    def __call__(self, gm: fx.GraphModule) -> Callable:
        if self.dump_graphs:
            graph_folder = dump_graphs_to_files(
                {"before_transformation_gm": gm.print_readable(False)}
            )
            enable_graph_optimization_dump(graph_folder)

        iter_gm = IterGraphModule(gm, enable_inductor=self.enable_inductor)
        if self.enable_graph_optimization:
            comm_fusion_with_concat(iter_gm, 100)
            schedule_comm_wait(iter_gm)
            remove_copy_from_optimizer(iter_gm)
        # Must be called after we are not going to move the graphs
        iter_gm.finalize_setup()

        if self.dump_graphs:
            dump_graphs_to_files(
                {
                    "iter_graph_setup_gm": iter_gm.setup_gm.print_readable(False),
                    "iter_graph_main_gm": iter_gm.main_gm.print_readable(False),
                    "iter_graph_cleanup_gm": iter_gm.cleanup_gm.print_readable(False),
                },
                graph_folder,
            )

        return iter_gm
