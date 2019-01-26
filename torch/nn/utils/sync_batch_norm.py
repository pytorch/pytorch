import torch

def convert_sync_batchnorm(module, process_group=None):
    r"""Helper function to convert `torch.nn.BatchNormND` layer in the model to
    `torch.nn.SyncBatchNorm` layer.

    Args:
        module (nn.Module): containing module
        process_group (optional): process group to scope synchronization,
    default is the whole world

    Returns:
        The original module with the converted `torch.nn.SyncBatchNorm` layer 

    Example::

        >>> # Network with nn.BatchNorm layer
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100)
        >>>          ).cuda()
        >>> # creating process group (optional)
        >>> # process_ids is a list of int identifying rank ids.
        >>> process_group = torch.distributed.new_group(process_ids)
        >>> sync_bn_module = convert_sync_batchnorm(module, process_group)

    """
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = torch.nn.SyncBatchNorm(module.num_features,
                                     module.eps, module.momentum,
                                     module.affine,
                                     module.track_running_stats,
                                     process_group)
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        mod.add_module(name, convert_sync_batchnorm(child))
    del module
    return mod
