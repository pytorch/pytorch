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
import numbers
import warnings
from torch._utils import _range
from torch._six import string_classes


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


def export(model, args, f, export_params=True, verbose=False, training=False,
           input_names=None, output_names=None, aten=False):
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
        input_names(list of strings, default empty list): names to assign to the
            input nodes of the graph, in order
        output_names(list of strings, default empty list): names to assign to the
            output nodes of the graph, in order
        aten (bool, default False): export the model in aten mode. If using aten mode,
            all the ops original exported by the functions in symbolic.py are exported
            as ATen ops.
    """
    _export(model, args, f, export_params, verbose, training, input_names, output_names)


def _optimize_trace(trace, aten):
    torch._C._jit_pass_peephole(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_onnx(trace, aten)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_onnx_peephole(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_dce(trace)
    torch._C._jit_pass_lint(trace)
    torch._C._jit_pass_canonicalize(trace)
    torch._C._jit_pass_lint(trace)


def _trace(func, args, return_outs=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    trace, torch_out = torch.jit.trace(func, args)
    _optimize_trace(trace)
    if return_outs:
        return trace, torch_out
    return trace


def _export(model, args, f, export_params=True, verbose=False, training=False,
            input_names=None, output_names=None, aten=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )

    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = model.state_dict().keys()

    # By default, training=False, which is good because running a model in
    # training mode could result in internal buffers getting updated, dropout
    # getting applied, etc.  If you really know what you're doing, you
    # can turn training=True (or None, to preserve whatever the original
    # training mode was.)
    with set_training(model, training):
        trace, torch_out = torch.jit.trace(model, args)

    if orig_state_dict_keys != model.state_dict().keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    _optimize_trace(trace, aten)

    _set_input_and_output_names(trace.graph(), input_names, output_names)

    if verbose:
        print(trace)

    # TODO: Don't allocate a in-memory string for the protobuf
    from torch.onnx.symbolic import _onnx_opset_version
    if export_params:
        # NB: OrderedDict values is not actually a list, but trace.export is
        # not duck-typed and expects an actual list.
        proto = trace.export(list(model.state_dict().values()), _onnx_opset_version)
    else:
        proto = trace.export([], _onnx_opset_version)

    torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
    return torch_out


def _set_input_and_output_names(graph, input_names, output_names):
    def set_names(node_list, name_list, descriptor):
        if name_list is None:
            return
        if len(name_list) != len(node_list):
            raise RuntimeError(
                "number of %s names provided (%d) did not match number of %ss (%d)"
                % (descriptor, len(name_list), descriptor, len(node_list)))
        for name, node in zip(name_list, node_list):
            node.setUniqueName(name)
    set_names(list(graph.inputs()), input_names, 'input')
    set_names(list(graph.outputs()), output_names, 'output')

attr_pattern = re.compile("^(.+)_([ifstgz])$")


def _run_symbolic_method(op_name, symbolic_fn, args):
    """
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        return symbolic_fn(*args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


def _is_onnx_list(value):
    if not isinstance(value, string_classes) and not torch.is_tensor(value) and isinstance(value, collections.Iterable):
        return True
    return False


def _add_attribute(node, key, value, aten):
    """ initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"
    if aten:
        if torch.is_tensor(value):
            # Caffe2 proto does not support tensor attribute.
            if value.numel() > 1:
                raise ValueError("Should not pass tensor attribute")
            value = _scalar(value)
            if isinstance(value, float):
                kind = "f"
            else:
                kind = "i"
    return getattr(node, kind + "_")(name, value)


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


def _newNode(g, opname, outputs, *args, **kwargs):
    aten = kwargs.pop("aten", False)
    n = g.create(opname, args, outputs)
    for k, v in sorted(kwargs.items()):
        # TODO: enable inplace in aten exporting mode.
        if k == "inplace":
            continue
        _add_attribute(n, k, v, aten=aten)
    return n


def _graph_op(g, opname, *raw_args, **kwargs):
    """
    Create an ONNX operator 'opname', taking 'args' as inputs and attributes
    'kwargs'; returning the node representing the single output of this operator
    (see the `outputs` keyword argument for multi-return nodes).

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.

    Arguments:
        opname (string): The ONNX operator name, e.g., `Abs` or `Add`.
        args (Node...): The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        kwargs: The attributes of the ONNX operator, with keys named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).
        outputs (int, optional):  The number of outputs this operator returns;
            by default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the ONNX operator
            in positional.
    """
    outputs = kwargs.pop('outputs', 1)

    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)

    def const_if_tensor(arg):
        if isinstance(arg, torch._C.Value):
            return arg
        else:
            return g.op("Constant", value_z=arg)

    args = list(const_if_tensor(arg) for arg in raw_args)
    n = g.appendNode(_newNode(g, opname, outputs, *args, **kwargs))
    if outputs == 1:
        return n.output()
    return tuple(o for o in n.outputs())


# Note [Export inplace]
# ~~~~~~~~~~~~~~~~~~~~~
# In abstract, it would be better for us to export inplace annotations,
# than to not export them, since it is useful information that can
# help the target of an ONNX export export more efficiently.  However,
# ONNX doesn't currently formalize inplace.  Fortunately, it's sound to drop
# inplace annotations, but we are losing information this way.


def _run_symbolic_function(g, n, inputs, aten=False):
    import torch.onnx.symbolic

    try:
        # See Note [Export inplace]
        if n.kind().endswith('_'):
            op_name = n.kind()[:-1]
        else:
            op_name = n.kind()
        # Export ops in aten mode.
        if aten:
            attrs = {k + "_" + n.kindOf(k)[0]: n[k] for k in n.attributeNames()}
            outputs = n.outputsSize()
            attrs["outputs"] = outputs
            return _graph_at(g, op_name, *inputs, aten=True, **attrs)

        # Export ONNX regular ops.
        attrs = {k: n[k] for k in n.attributeNames()}
        if not hasattr(torch.onnx.symbolic, op_name):
            warnings.warn("ONNX export failed on {} because torch.onnx.symbolic.{} does not exist"
                          .format(op_name, op_name))
            return None
        fn = getattr(torch.onnx.symbolic, op_name)
        return fn(g, *inputs, **attrs)

    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


# Generate an ONNX ATen op node.
def _graph_at(g, opname, *args, **kwargs):
    return g.op("ATen", *args, operator_s=opname, **kwargs)


# This helper function can create either constant tensor or constant scalar.
# If dims is None or 0 or [0], generate a 0-d tensor (scalar).
#
# TODO: We might not need this anymore, since most scalars now show up
# as tensors
def _graph_constant(g, value, dims, type, *args, **kwargs):
    assert isinstance(value, numbers.Number)
    assert type is not None
    isscalar = False
    if dims is None or dims == 0 or set(dims) == set([0]):
        dims = [1]
        isscalar = True
    type = type.lower()
    if type == "char":
        tensor = torch.CharTensor(*dims)
    elif type == "short":
        tensor = torch.ShortTensor(*dims)
    elif type == "int":
        tensor = torch.IntTensor(*dims)
    elif type == "long":
        tensor = torch.LongTensor(*dims)
    elif type == "half":
        tensor = torch.HalfTensor(*dims)
    elif type == "float":
        tensor = torch.FloatTensor(*dims)
    elif type == "double":
        tensor = torch.DoubleTensor(*dims)
    else:
        raise ValueError("Unknown type, type should be one of the following strings: "
                         "char, short, int, long, half, float, double")
    tensor.fill_(value)
    if isscalar:
        return g.op("Constant", *args, value_z=tensor, **kwargs)
    return g.op("Constant", *args, value_t=tensor, **kwargs)


def _node_getitem(self, k):
    """
    Accessor for attributes of a node which is polymorphic over
    return type.

    NB: This is monkey-patched onto Node.
    """
    sel = self.kindOf(k)
    return getattr(self, sel)(k)


torch._C.Graph.op = _graph_op
torch._C.Graph.at = _graph_at
torch._C.Graph.constant = _graph_constant
torch._C.Node.__getitem__ = _node_getitem
