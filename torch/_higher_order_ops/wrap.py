from torch._ops import HigherOrderOperator
from torch.utils.checkpoint import checkpoint
from itertools import count
import inspect

uid = count(1)

# Used for testing the HigherOrderOperator mechanism
class Wrap(HigherOrderOperator):
    def __init__(self):
        super().__init__("wrap")

    def __call__(self, func, *args, **kwargs):
        # Dynamo already traces the body of HigherOrderOp beforehand when it
        # so no need to trace into it.
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        @disable
        def wrapper():
            result = func(*args, **kwargs)
            return result

        return wrapper()

wrap = Wrap()

class WrapActivationCheckpoint(HigherOrderOperator):
    """
    This operator is used to wrap torch.utils.checkpoint. This avoids
    TorchDynamo to look into saved tensor hooks and directly passes the control
    to AOT Autograd, which is ok with tracing saved tensor hooks. As a result of
    AOT tracing torch.utils.checkpoint code, we have a backward graph with
    recomputed forward nodes.

    However, we might deprecate this operator soon. The difficulty arises in the
    functionalization of rng ops. Today, there are two different
    functionalization of rng ops - one at AOT autograd and other at Inductor.
    And they are difficult to map to each other. The rng states also complicate
    pattern matching in Inductor. Due to the ease of implementation, we are
    currently inclined towards functionalization at Inductor level, which means
    that duplication/recomputation is done as a compiler pass in the
    partitioners. See TagActivationCheckpoint for more information.
    """
    def __init__(self):
        super().__init__("wrap_activation_checkpoint")

    def __call__(self, function, *args, **kwargs):
        # use_reentrant is set to False because this op is going to be traced.
        # And we ensure that AOT Autograd traces through the non reentrant
        # version of checkpointing.
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter
        kwargs["use_reentrant"] = False
        kwargs["preserve_rng_state"] = False
        # Using interpreter allows preservation of metadata through torch.compile stack.
        with fx_traceback.preserve_node_meta():
            return checkpoint(Interpreter(function).run, *args, **kwargs)

wrap_activation_checkpoint = WrapActivationCheckpoint()

class TagActivationCheckpoint(HigherOrderOperator):
    """
    This operator is supposed to be used only with torch.compile stack. This
    accepts a Fx graph module which needs to be checkpointed. This operator adds
    "recomputable" tag to the nodes of the Fx graph that should be recomputed.

    The goal is to avoid both Dynamo and AOT Autograd to trace through saved
    tensor hooks, and rather rely on the partitioners to actually duplicate the
    nodes. This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops. Therefore, the duplication of nodes, by design, respects the rng states
    in the forward and recomputed forward in backward.
    """

    def __init__(self):
        super().__init__("tag_activation_checkpoint")

    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
        ckpt_signature = inspect.signature(checkpoint)
        checkpoint_keys = set()
        for name in ckpt_signature.parameters:
            if name in ("function", "args", "kwargs"):
                continue
            checkpoint_keys.add(name)

        # `preserve_rng_state` is not a regular kwarg
        checkpoint_keys.add("preserve_rng_state")

        checkpoint_kwargs = {name: kwargs[name] for name in kwargs.keys() if name in checkpoint_keys}
        gmod_kwargs = {name: kwargs[name] for name in kwargs.keys() if name not in checkpoint_keys}
        return checkpoint_kwargs, gmod_kwargs

    def tag_nodes(self, gmod):
        # TODO - This needs major investigation. Currently, we are tagging all
        # the forward nodes as recomputable. However, torch.utils.checkpoint
        # provides a custom function to selectively recompute. We will have to
        # figure out how to tag seletively.
        unique_graph_id = next(uid)
        for node in gmod.graph.nodes:
            if node.op in ("call_function", "call_method", "call_module"):
                node.meta["recompute"] = unique_graph_id
        return gmod

    def __call__(self, gmod, *args, **kwargs):
        if "context_fn" in kwargs:
            raise RuntimeError("Tagged Activation checkpointing does not support selective checkpointing yet.")
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter
        gmod = self.tag_nodes(gmod)
        # Using interpreter allows preservation of metadata through torch.compile stack.
        with fx_traceback.preserve_node_meta():
            return Interpreter(gmod).run(*args)

tag_activation_checkpoint = TagActivationCheckpoint()
