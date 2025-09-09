import os
import sys
from typing import Optional

from torch.utils._config_module import install_config_module


# Whether to use c10d._time_estimator for collectives runtime estimations.
runtime_estimations_use_nccl_lib_estimations: bool = False

# Config to enable sync of runtime estimations across distributed ranks,
# To prevent passes using this runtime estimations to make different
# decisions on different distributed ranks.
runtime_estimations_align_across_all_distributed_ranks: bool = False

reorder_iterative_debug_memory_recompute: bool = False
reorder_iterative_debug_limit_to_reorder: Optional[int] = (
    None
    if (env_str := os.getenv("PYTORCH_REORDER_COLLECTIVES_LIMIT")) is None
    else int(env_str)
)
sink_waits_iterative_debug_limit_to_sink: Optional[int] = (
    None if (env_str := os.getenv("PYTORCH_SINK_WAITS_LIMIT")) is None else int(env_str)
)


# Should be used with config.runtime_estimations_mms_benchmark = True
reorder_iterative_use_runtime_estimations: bool = False
sink_iterative_use_runtime_estimations: bool = False
reorder_for_compute_comm_overlap_broadcast_runtime_estimations: bool = False

reorder_sink_runtime_estimations_comm_mult: float = 2.0
reorder_sink_runtime_estimations_non_comm_mult: float = 1.0
# Ratio of comm_time to cover deviations of comm_time from estimations
reorder_iterative_extra_comm_comp_overlap: float = 0.5
sink_iterative_extra_comm_comp_overlap: float = 0.5
reorder_iterative_peak_memory_budget: float = 0.2
sink_iterative_peak_memory_budget: float = 0.2

# Experimental unsafe configuration that allows changing relative collectives order.
# Must be used with runtime_estimations_align_across_all_distributed_ranks = True
reorder_iterative_unsafe_collectives_reorder: bool = True
sink_waits_iterative_unsafe_collectives_reorder: bool = True

# Allow group and move other collectives during reordering
reorder_iterative_group_with_collectives: bool = False
sink_waits_iterative_swap_with_collectives: bool = False

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
