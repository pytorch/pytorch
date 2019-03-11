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
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.SyncBatchNorm(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats,
                                               process_group)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child))
    del module
    return module_output
