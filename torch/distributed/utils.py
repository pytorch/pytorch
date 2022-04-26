import torch.distributed as dist

def _verify_param_shape_across_processes(process_group, tensors, logger=None):
    return dist._verify_params_across_processes(process_group, tensors, logger)

def _sync_params_and_buffers(
    module,
    process_group,
    broadcast_bucket_size,
    rank,
    params_and_buffers_to_ignore,
):
    """
    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``verify_param_shape_across_processes``.
    """
    module_states = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    for name, buffer in module.named_buffers():
        if name not in params_and_buffers_to_ignore:
            module_states.append(buffer.detach())

    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, rank
        )
