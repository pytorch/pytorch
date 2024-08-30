import torch.fx


class BackwardState:
    """
    BackwardState is used to pass Python hooks from the forwards pass
    into the backwards pass in Dynamo+Compiled Autograd.

    It is created by TorchDynamo and has special handling there.
    Dynamo will pass an empty BackwardState to the forwards, then populate
    members on it (via setattr) only after the forwards graph is finished.
    Later on, in CompileAutograd we will inline and add the needed guards
    on the BackwardState.

    BackwardState is identified and has special handling in AOTAutograd.
    During AOTAutograd:
        1) BackwardState is an input to the forwards graph
        2) It must only be used in the backwards
        3) It will be empty in the forwards
        4) In the forwards we add a wrapper to save it
        5) In the backwards it becomes an input
        6) There can only be one per graph

    BackwardState requires CompiledAutograd.
    """

    proxy: torch.fx.Proxy
    # BackwardState object is a singleton shared between all graphs, so that
    # module forward hooks captured during Dynamo tracing of the forward pass
    # can be accessed during Dynamo tracing of the compiled autograd graph.
    # Otherwise, BackwardState object created for each graph will be different,
    # causing the module forward hooks captured during forward pass to be lost.
    # (This is particularly important for activation checkpointing where we need to
    # rerun module forward hooks in backward pass.)
    singleton = None

    @classmethod
    def get_singleton(cls):
        if cls.singleton is None:
            cls.singleton = BackwardState()
        return cls.singleton
