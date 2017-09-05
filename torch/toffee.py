"""
The torch.toffee module contains functions to export models into the Toffee
IR format.  These models can be loaded with the ToffeeIR library and then
converted to models which run on other deep learning frameworks.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
import collections
from ._utils import _range


def export(model, args, f, export_params=True, kwargs=None, verbose=False):
    """
    Export a model into Toffee format.  This exporter runs your model
    once in order to get a trace of its execution to be exported; at the
    moment, it does not support dynamic models (e.g., RNNs.)

    See also: :ref:`toffee-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (torch.autograd.Variable or tuple of variables): the inputs to
            the model, e.g., such that ``model(*args, **kwargs)`` is a valid
            invocation of the model (see kwargs below).
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you are exporting an
            untrained model.
        kwargs (dict, optional): keyword inputs to the model.
    """
    _export(model, args, f, export_params, kwargs, verbose)


def _export(model, args, f, export_params=True, kwargs=None, verbose=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )
    if not kwargs:
        kwargs = {}
    trace, torch_out = torch.jit.record_trace(model, *args, **kwargs)
    # TODO: Don't allocate a in-memory string for the protobuf
    if export_params:
        proto = trace.export(model.state_dict().values(), verbose)
    else:
        proto = trace.export(verbose)
    torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
    return torch_out


attr_pattern = re.compile("^(.+)_([ifstg])$")


def _add_attribute(node, key, value):
    """ initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if isinstance(value, collections.Iterable):
        kind += "s"
    return getattr(node, kind + '_')(name, value)


def _newNode(self, opname, *args, **kwargs):
    n = self.create(opname, args)
    for k, v in sorted(kwargs.items()):
        _add_attribute(n, k, v)
    return n


def _op(self, opname, *args, **kwargs):
    outputs = kwargs.pop('outputs', 1)
    n = self.appendNode(_newNode(self, opname, *args, **kwargs))
    if outputs == 1:
        return n
    return tuple(self.appendNode(self.createSelect(n, i)) for i in _range(outputs))

torch._C.Graph.op = _op
