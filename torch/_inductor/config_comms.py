import sys

from torch.utils._config_module import install_config_module


# Whether to use c10d._time_estimator for collectives runtime estimations.
runtime_estimations_use_nccl_lib_estimations: bool = False

# Config to enable sync of runtime estimations across distributed ranks,
# To prevent passes using this runtime estimations to make different
# decisions on different distributed ranks.
runtime_estimations_align_across_all_distributed_ranks: bool = False

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
