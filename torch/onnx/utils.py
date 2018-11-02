r"""
The torch.onnx module contains functions to export models into the ONNX
IR format.  These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
from torch._six import container_abcs
import contextlib
import numbers
import warnings
import functools
import types
from torch._six import string_classes
from torch.autograd import Function, function
from torch.jit import _unique_state_dict
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
from torch._C import ListType


@contextlib.contextmanager
def set_training(model, mode):
    r"""
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
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None):
    r"""
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported;
    at the moment, it supports a limited set of dynamic models (e.g., RNNs.)

    See also: :ref:`onnx-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (tuple of arguments): the inputs to
            the model, e.g., such that ``model(*args)`` is a valid
            invocation of the model.  Any non-Tensor arguments will
            be hard-coded into the exported model; any Tensor arguments
            will become inputs of the exported model, in the order they
            occur in args.  If args is a Tensor, this is equivalent
            to having called it with a 1-ary tuple of that Tensor.
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
        aten (bool, default False): [DEPRECATED. use operator_export_type] export the
            model in aten mode. If using aten mode, all the ops original exported
            by the functions in symbolic.py are exported as ATen ops.
        export_raw_ir (bool, default False): [DEPRECATED. use operator_export_type]
            export the internal IR directly instead of converting it to ONNX ops.
        operator_export_type (enum, default OperatorExportTypes.ONNX):
            OperatorExportTypes.ONNX: all ops are exported as regular ONNX ops.
            OperatorExportTypes.ONNX_ATEN: all ops are exported as ATen ops.
            OperatorExportTypes.ONNX_ATEN_FALLBACK: if symbolic is missing,
                                                    fall back on ATen op.
            OperatorExportTypes.RAW: export raw ir.
    """
    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ATEN if aten else OperatorExportTypes.RAW
    elif operator_export_type is None:
        if torch.onnx.PYTORCH_ONNX_CAFFE2_BUNDLE:
            operator_export_type = OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = OperatorExportTypes.ONNX
    _export(model, args, f, export_params, verbose, training, input_names, output_names,
            operator_export_type=operator_export_type)


# ONNX can't handle constants that are lists of tensors, which can
# get generated in constant prop. So we split them back into prim::ListConstructs
def _split_tensor_list_constants(g, block):
    for node in block.nodes():
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        if node.kind() == "prim::Constant":
            output_type = node.output().type()
            if output_type.isSubtypeOf(ListType.ofTensors()):
                inputs = [g.create("prim::Constant").t_('value', t)
                           .insertBefore(node).output()
                          for t in node['value']]
                lc = (g.create("prim::ListConstruct", inputs)
                      .insertBefore(node)
                      .output()
                      .setType(ListType.ofTensors()))
                node.output().replaceAllUsesWith(lc)


def _optimize_graph(graph, operator_export_type):
    torch._C._jit_pass_remove_inplace_ops(graph)
    # we record now record some ops like ones/zeros
    # into a trace where we previously recorded constants
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    torch._C._jit_pass_constant_propagation(graph)
    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_canonicalize_ops(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    torch._C._jit_pass_prepare_division_for_onnx(graph)
    # onnx only supports tensors, so we turn all out number types into tensors
    torch._C._jit_pass_erase_number_types(graph)
    # onnx does not support tuples, so try to remove them
    torch._C._jit_pass_lower_all_tuples(graph)
    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    if operator_export_type != OperatorExportTypes.RAW:
        graph = torch._C._jit_pass_onnx(graph, operator_export_type)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_onnx_peephole(graph)
        torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_fixup_onnx_loops(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


def _trace(func, args, operator_export_type, return_outs=False):
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args, )

    trace, torch_out = torch.jit.get_trace_graph(func, args)
    trace.set_graph(_optimize_graph(trace.graph(), operator_export_type))
    if return_outs:
        return trace, torch_out
    return trace


def _trace_and_get_graph_from_model(model, args, training):

    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = _unique_state_dict(model).keys()

    # By default, training=False, which is good because running a model in
    # training mode could result in internal buffers getting updated, dropout
    # getting applied, etc.  If you really know what you're doing, you
    # can turn training=True (or None, to preserve whatever the original
    # training mode was.)
    with set_training(model, training):
        trace, torch_out = torch.jit.get_trace_graph(model, args)

    if orig_state_dict_keys != _unique_state_dict(model).keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    return trace.graph(), torch_out


def _model_to_graph(model, args, f, verbose=False, training=False,
                    input_names=None, output_names=None,
                    operator_export_type=OperatorExportTypes.ONNX,
                    example_outputs=None, propagate=False):
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args, )

    if isinstance(model, torch.jit.ScriptModule):
        torch_out = None
        assert example_outputs is not None, "example_outputs must be provided when exporting a ScriptModule"
        if isinstance(example_outputs, torch.Tensor):
            example_outputs = [example_outputs]
        try:
            method = model.__getattr__('forward')
            graph = method.propagate_and_assign_input_and_output_shapes(
                args, example_outputs, False, propagate)
            # Erase number types to bring the graph to a pre-NumberType state
            params = method.params()
        except AttributeError:
            # TODO: just trace it
            raise RuntimeError('\'forward\' method must be a script method')
    else:
        graph, torch_out = _trace_and_get_graph_from_model(model, args, training)
        params = list(_unique_state_dict(model).values())

    graph = _optimize_graph(graph, operator_export_type)

    # NB: ONNX requires complete information about output types, which might be
    # erased by some optimizations, so we need to set it explicitly again.
    if torch_out is not None:
        output_tensors, _ = torch._C._jit_flatten(torch_out)
        for output, tensor in zip(graph.outputs(), output_tensors):
            output.inferTypeFrom(tensor)

    _set_input_and_output_names(graph, input_names, output_names)
    if verbose:
        print(graph)

    return graph, params, torch_out


def export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False,
                            input_names=None, output_names=None, aten=False, export_raw_ir=False,
                            operator_export_type=None, export_type=ExportTypes.PROTOBUF_FILE,
                            example_outputs=None, propagate=False, google_printer=False):
    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ATEN if aten else OperatorExportTypes.RAW
    elif operator_export_type is None:
        operator_export_type = OperatorExportTypes.ONNX
    return _export_to_pretty_string(model, args, f, export_params, verbose, training,
                                    input_names, output_names, operator_export_type,
                                    export_type, example_outputs, propagate, google_printer)


def _export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False,
                             input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX,
                             export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False,
                             google_printer=False):
    graph, params, torch_out = _model_to_graph(model, args, f, verbose,
                                               training, input_names,
                                               output_names, operator_export_type,
                                               example_outputs, propagate)

    from torch.onnx.symbolic import _onnx_opset_version
    return graph.prettyPrintExport(params, _onnx_opset_version, False, operator_export_type, google_printer)


# NOTE: the output `torch_out` will contain the output tensors resulting from
# the trace of a Module. In the case that a torch.nn.ScriptModule is passed in,
# this output will be None, since we are not doing any tracing but rather
# directly extracting the graph.
def _export(model, args, f, export_params=True, verbose=False, training=False,
            input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX,
            export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False):
    graph, params, torch_out = _model_to_graph(model, args, f, verbose,
                                               training, input_names,
                                               output_names, operator_export_type,
                                               example_outputs, propagate)

    # TODO: Don't allocate a in-memory string for the protobuf
    from torch.onnx.symbolic import _onnx_opset_version
    defer_weight_export = export_type is not ExportTypes.PROTOBUF_FILE
    if export_params:
        proto, export_map = graph.export(params, _onnx_opset_version, defer_weight_export, operator_export_type)
    else:
        proto, export_map = graph.export([], _onnx_opset_version, False, operator_export_type)

    if export_type == ExportTypes.PROTOBUF_FILE:
        assert(len(export_map) == 0)
        torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
    elif export_type in [ExportTypes.ZIP_ARCHIVE, ExportTypes.COMPRESSED_ZIP_ARCHIVE]:
        import zipfile
        compression = zipfile.ZIP_DEFLATED \
            if export_type == ExportTypes.COMPRESSED_ZIP_ARCHIVE \
            else zipfile.ZIP_STORED
        with zipfile.ZipFile(f, 'w', compression=compression) as z:
            z.writestr(ONNX_ARCHIVE_MODEL_PROTO_NAME, proto)
            for k, v in export_map.items():
                z.writestr(k, v)
    elif export_type == ExportTypes.DIRECTORY:
        import os
        if os.path.exists(f):
            assert(os.path.isdir(f))
        else:
            os.makedirs(f)

        model_proto_file = os.path.join(f, ONNX_ARCHIVE_MODEL_PROTO_NAME)
        torch.serialization._with_file_like(
            model_proto_file, "wb", lambda f: f.write(proto))

        for k, v in export_map.items():
            weight_proto_file = os.path.join(f, k)
            torch.serialization._with_file_like(
                weight_proto_file, "wb", lambda f: f.write(v))
    else:
        raise RuntimeError('Unknown export type')
    return torch_out


def _set_input_and_output_names(graph, input_names, output_names):
    def set_names(node_list, name_list, descriptor):
        if name_list is None:
            return
        if len(name_list) > len(node_list):
            raise RuntimeError(
                "number of %s names provided (%d) exceeded number of %ss (%d)"
                % (descriptor, len(name_list), descriptor, len(node_list)))
        for name, node in zip(name_list, node_list):
            if node.uniqueName() != name:
                node.setUniqueName(name)
    set_names(list(graph.inputs()), input_names, 'input')
    set_names(list(graph.outputs()), output_names, 'output')

attr_pattern = re.compile("^(.+)_([ifstgz])$")


def _run_symbolic_method(op_name, symbolic_fn, args):
    r"""
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
    if not isinstance(value, string_classes) and \
            not isinstance(value, torch.Tensor) and \
            isinstance(value, container_abcs.Iterable):
        return True
    return False


def _add_attribute(node, key, value, aten):
    r""" initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"
    if aten:
        if isinstance(value, torch.Tensor):
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
    if "::" in opname:
        aten = False
        ns_opname = opname
    else:
        aten = kwargs.pop("aten", False)
        ns = "aten" if aten else "onnx"
        ns_opname = ns + "::" + opname
    n = g.create(ns_opname, args, outputs)
    for k, v in sorted(kwargs.items()):
        # TODO: enable inplace in aten exporting mode.
        if k == "inplace":
            continue
        _add_attribute(n, k, v, aten=aten)
    return n


def _graph_op(g, opname, *raw_args, **kwargs):
    r"""
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
        if arg is None:
            return arg
        elif isinstance(arg, torch._C.Value):
            return arg
        else:
            return g.op("Constant", value_z=arg)

    args = list(const_if_tensor(arg) for arg in raw_args)
    n = g.insertNode(_newNode(g, opname, outputs, *args, **kwargs))
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


def _run_symbolic_function(g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX):
    # NB: Returning None means the node gets cloned as is into
    # the new graph
    try:
        import torch.onnx.symbolic

        # See Note [Export inplace]
        # TODO: I think this is not necessary anymore
        if n.kind().endswith('_'):
            ns_op_name = n.kind()[:-1]
        else:
            ns_op_name = n.kind()
        ns, op_name = ns_op_name.split("::")

        if ns == "onnx":
            # Use the original node directly
            return None

        elif ns == "aten":
            is_exportable_aten_op = hasattr(torch.onnx.symbolic, op_name)
            is_onnx_aten_export = operator_export_type == OperatorExportTypes.ONNX_ATEN
            is_aten_fallback_export = operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK
            if is_onnx_aten_export or (not is_exportable_aten_op and is_aten_fallback_export):
                # Direct ATen export requested
                attrs = {k + "_" + n.kindOf(k)[0]: n[k] for k in n.attributeNames()}
                outputs = n.outputsSize()
                attrs["outputs"] = outputs
                return _graph_at(g, op_name, *inputs, aten=True, **attrs)

            else:
                # Export it regularly
                attrs = {k: n[k] for k in n.attributeNames()}
                if not is_exportable_aten_op:
                    warnings.warn("ONNX export failed on ATen operator {} because torch.onnx.symbolic.{} does not exist"
                                  .format(op_name, op_name))
                    return None
                fn = getattr(torch.onnx.symbolic, op_name)
                return fn(g, *inputs, **attrs)

        elif ns == "prim":
            if op_name == "Constant":
                if n.kindOf("value") == "t":
                    return g.op("Constant", value_t=n["value"])
                elif n.kindOf("value") == "is":
                    value = torch.stack([torch.tensor(v) for v in n["value"]]) if n["value"] else []
                    return g.op("Constant", value_t=value)
                else:
                    raise RuntimeError("Unsupported prim::Constant kind: `{}`. Send a bug report.".format(
                        n.kindOf("value")))
            elif op_name == "Undefined" or op_name == "None" or op_name == "ListConstruct":
                # Undefined/None is not an ONNX operator; keep it as prim::Undefined/
                # prim::None and let the exporter handle finally eliminating these

                # For ListConstruct, it will be erased in the ONNX peephole pass
                return None
            elif op_name == 'Loop' or op_name == 'If':
                new_op_outputs = g.op(op_name, *inputs, outputs=n.outputsSize())
                new_node = new_op_outputs[0].node() if n.outputsSize() > 1 else new_op_outputs.node()
                for b in n.blocks():
                    new_block = new_node.addBlock()
                    torch._C._jit_pass_onnx_block(b, new_block, operator_export_type, env)
                return new_op_outputs
            else:
                symbolic_name = 'prim_' + op_name
                symbolic_fn = getattr(torch.onnx.symbolic, symbolic_name, None)
                if symbolic_fn is None:
                    warnings.warn("ONNX export failed on primitive operator {}; please report a bug".format(op_name))
                    return None
                attrs = {k: n[k] for k in n.attributeNames()}
                return symbolic_fn(g, *inputs, **attrs)

        else:
            warnings.warn("ONNX export failed on an operator with unrecognized namespace {}::{}; "
                          "please report a bug".format(ns, op_name))
            return None

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
    r"""
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
