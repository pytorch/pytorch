import torch.nn as nn
from torch._functorch.utils import exposed_in


def batch_norm_without_running_stats(module: nn.Module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        module.track_running_stats = False


@exposed_in("torch.func")
def replace_all_batch_norm_modules_(root: nn.Module) -> nn.Module:
    """
    In place updates :attr:`root` by setting the ``running_mean`` and ``running_var`` to be None and
    setting track_running_stats to be False for any nn.BatchNorm module in :attr:`root`
    """
    # base case
    batch_norm_without_running_stats(root)

    for obj in root.modules():
        batch_norm_without_running_stats(obj)
    return root
