from torch.nn.parallel import DistributedDataParallel as DDP


def basic_ddp_model(self, rank, model, process_group, hook_state, hook):
    r"""
    A function that creates a ddp_model and hook_state objects.
    The ddp model is is initialized with a single device id and
    the process group. The ddp_model also registers the communication
    hook.
    Args:
        rank (int): worker rank
        model (nn.Module): neural network model
        process_group (ProcessGroup): distributed process group
        HookState (class): class that will be used to keep track of state
            during training.
        hook (function): ddp communication hook
    """
    ddp_model = DDP(
        model, device_ids=[rank], process_group=process_group
    )
    hook_state = hook_state(self, process_group)
    ddp_model.register_comm_hook(hook_state, hook)
    return ddp_model, hook_state
