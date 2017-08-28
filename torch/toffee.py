import torch


def op(op_name, *args, **kwargs):
    """A primitive operator

    TODO: better docs here

    TODO: This doesn't actually do an operation, eventually we want it to (and
    trace correctly).  DO NOT rely on this returning a dictionary!!!
    """
    return {"name": op_name, "inputs": args, "attrs": kwargs}


def export(model, input, embed_params):

    # Enable tracing on the model
    trace, torch_out = torch.jit.record_trace(model, input)
    if embed_params is False:
        proto = trace.export()
    else:
        proto = trace.export(model.state_dict().values())
    # TODO: a way to print the proto
    return proto, torch_out
