"""
The torch.onnx module contains functions to export models into the ONNX
IR format.  These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
import collections
import string
import json
import math
import contextlib
from ._utils import _range


@contextlib.contextmanager
def set_training(model, mode):
    """
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.
    """
    if mode is None:
        yield
        return
    old_mode = model.training
    if old_mode != mode:
        model.train(mode)
    try:
        yield
    finally:
        if old_mode != mode:
            model.train(old_mode)


def export(model, args, f, export_params=True, verbose=False, training=False):
    """
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported; at the
    moment, it does not support dynamic models (e.g., RNNs.)

    See also: :ref:`onnx-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (tuple of arguments): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid
            invocation of the model.  Any non-Variable arguments will
            be hard-coded into the exported model; any Variable arguments
            will become inputs of the exported model, in the order they
            occur in args.  If args is a Variable, this is equivalent
            to having called it with a 1-ary tuple of that Variable.
            (Note: passing keyword arguments to the model is not currently
            supported.  Give us a shout if you need it.)
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, the ordering as specified by ``model.state_dict().values()``
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.
        training (bool, default False): export the model in training mode.  At
            the moment, ONNX is oriented towards exporting models for inference
            only, so you will generally not need to set this to True.
    """
    _export(model, args, f, export_params, verbose, training)


def _export(model, args, f, export_params=True, verbose=False, training=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )
    # Look at the state_dict *prior* to running the model, as this
    # accurately captures what inputs we actually passed to the model.
    # If we run it afterwards, a buggy forward pass could have
    # added/deleted parameters, changing the structure of the state_dict.
    #
    # Note that it's possible the actual /data/ may change by the time we
    # actually export the model, because we're not cloning the parameters
    # here.  This shouldn't happen, because we've turned off training
    # mode and networks in inference mode should be purely functional.
    # But it's user code; anything can happen!
    params = None
    if export_params:
        # NB: OrderedDict values is not actually a list, but trace.export is
        # not duck-typed and expects an actual list.
        params = list(model.state_dict().values())
    # It's important to run the model in inference mode when exporting;
    # otherwise internal buffers may get updated, dropout gets applied, etc.
    with set_training(model, training):
        trace, torch_out = torch.jit.record_trace(model, *args, num_derivatives=0)
    torch._C._jit_pass_onnx(trace)
    if verbose:
        print(trace)
    # TODO: Don't allocate a in-memory string for the protobuf
    if export_params:
        proto = trace.export(params)
    else:
        proto = trace.export()
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
