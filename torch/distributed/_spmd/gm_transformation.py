import operator
from typing import Any, Callable, Dict, List, Optional

from functorch import make_fx

import torch
import torch.nn as nn

from torch import fx
from torch.distributed._spmd.graph_optimization import (
    comm_fusion_with_concat,
    enable_graph_optimization_dump,
    remove_copy_from_optimizer,
    schedule_comm_wait,
)
from torch.distributed._spmd.graph_utils import dump_graphs_to_files
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.distributed._spmd.partial_lower import partial_lower


class GraphModuleTransformation:
    def __init__(
        self,
        num_iters: int,
        *,
        enable_graph_optimization: bool = False,
        enable_inductor: bool = False,
        dump_graphs: bool = False,
    ) -> None:
        self.num_iters = num_iters
        self.enable_graph_optimization = enable_graph_optimization
        self.enable_inductor = enable_inductor
        self.dump_graphs = dump_graphs

    def __call__(self, gm: fx.GraphModule) -> Callable:
        if self.dump_graphs:
            graph_folder = dump_graphs_to_files(
                {"before_transformation_gm": gm.print_readable(False)}
            )
            enable_graph_optimization_dump(graph_folder)

        iter_gm = IterGraphModule(gm)
        if self.enable_graph_optimization:
            comm_fusion_with_concat(iter_gm, 100)
            schedule_comm_wait(iter_gm)
            remove_copy_from_optimizer(iter_gm)
        iter_gm.freeze_cross_iter_movement()
        iter_gm.setup(self.num_iters)

        if self.dump_graphs:
            dump_graphs_to_files(
                {
                    "iter_graph_setup_gm": iter_gm.setup_gm.print_readable(False),
                    "iter_graph_main_gm": iter_gm.main_gm.print_readable(False),
                    "iter_graph_cleanup_gm": iter_gm.cleanup_gm.print_readable(False),
                },
                graph_folder,
            )

        if self.enable_inductor:
            iter_gm.main_gm = partial_lower(iter_gm.main_gm)

        return iter_gm
