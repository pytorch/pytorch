import torch


def export(model, input, embed_params):

    # Enable tracing on the model
    trace, torch_out = torch.jit.record_trace(model, input)
    if embed_params is False:
        proto = trace.export()
    else:
        proto = trace.export(model.state_dict().values())
    # TODO: a way to print the proto
    return proto, torch_out
