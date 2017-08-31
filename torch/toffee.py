"""
The torch.toffee module contains functions to export models into the Toffee
IR format.  These models can be loaded with the ToffeeIR library and then
converted to models which run on other deep learning frameworks.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization


def export(model, args, f, export_params=True, kwargs=None):
    """
    Export a model into Toffee format.  This exporter runs your model
    once in order to get a trace of its execution to be exported; at the
    moment, it does not support dynamic models (e.g., RNNs.)

    See also: :ref:`toffee-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (torch.autograd.Variable or tuple of variables): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid invocation
            of the model.
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you are exporting an
            untrained model.
        kwargs (dict, optional): keyword inputs to the model.
    """
    _export(model, args, f, export_params=export_params, kwargs=None)


# Internal helper function which also returns the computed tensors, which
# can be useful for comparing PyTorch's execution of the model with the
# eventual runner.
def _export(model, args, f, export_params=True, kwargs=None):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )
    if not kwargs:
        kwargs = {}
    trace, torch_out = torch.jit.record_trace(model, *args, **kwargs)
    # TODO: Don't allocate a in-memory string for the protobuf
    if export_params:
        proto = trace.export(model.state_dict().values())
    else:
        proto = trace.export()
    torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
    return torch_out
    # NB: It's very important that trace dies at the end of this function;
    # otherwise you can't retrace the model.
