from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint

# Used for testing the HigherOrderOperator mechanism
class Wrap(HigherOrderOperator):
    def __init__(self):
        super().__init__("wrap")

    def __call__(self, func, *args):
        result = func(*args)
        return result

wrap = Wrap()

class WrapActivationCheckpoint(HigherOrderOperator):
    """
    Wrap activation checkpoint is little different from Wrap. There are two
    functions to wrap here - utils.checkpoint() itself and the function to be
    checkpointed (the first arg to the utils.checkpoint() function).
    """
    def __init__(self):
        super().__init__("wrap_activation_checkpoint")

    def __call__(self, function, *args, **kwargs):
        # use_reentrant is set to False because this op is going to be traced.
        # And we ensure that AOT Autograd traces through the non reentrant
        # version of checkpointing.
        kwargs["use_reentrant"] = False
        result = checkpoint(function, *args, **kwargs)
        return result

wrap_activation_checkpoint = WrapActivationCheckpoint()
