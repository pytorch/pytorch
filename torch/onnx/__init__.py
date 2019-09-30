import torch._C as _C

TensorProtoDataType = _C._onnx.TensorProtoDataType
OperatorExportTypes = _C._onnx.OperatorExportTypes
PYTORCH_ONNX_CAFFE2_BUNDLE = _C._onnx.PYTORCH_ONNX_CAFFE2_BUNDLE

ONNX_ARCHIVE_MODEL_PROTO_NAME = "__MODEL_PROTO"

# TODO: Update these variables when there
# is a new ir_version and producer_version
# and use these values in the exporter
ir_version = 4
producer_name = "pytorch"
producer_version = "1.3"


class ExportTypes:
    PROTOBUF_FILE = 1
    ZIP_ARCHIVE = 2
    COMPRESSED_ZIP_ARCHIVE = 3
    DIRECTORY = 4


def _export(*args, **kwargs):
    from torch.onnx import utils
    result = utils._export(*args, **kwargs)
    return result


def export(model, args, f, export_params=True, verbose=False, training=False,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=None, _retain_param_name=True,
           do_constant_folding=False, example_outputs=None, strip_doc_string=True,
           dynamic_axes=None, keep_initializers_as_inputs=None):
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
            by the functions in symbolic_opset<version>.py are exported as ATen ops.
        export_raw_ir (bool, default False): [DEPRECATED. use operator_export_type]
            export the internal IR directly instead of converting it to ONNX ops.
        operator_export_type (enum, default OperatorExportTypes.ONNX):
            OperatorExportTypes.ONNX: all ops are exported as regular ONNX ops.
            OperatorExportTypes.ONNX_ATEN: all ops are exported as ATen ops.
            OperatorExportTypes.ONNX_ATEN_FALLBACK: if symbolic is missing,
                                                    fall back on ATen op.
            OperatorExportTypes.RAW: export raw ir.
        opset_version (int, default is 9): by default we export the model to the
            opset version of the onnx submodule. Since ONNX's latest opset may
            evolve before next stable release, by default we export to one stable
            opset version. Right now, supported stable opset version is 9.
            The opset_version must be _onnx_master_opset or in _onnx_stable_opsets
            which are defined in torch/onnx/symbolic_helper.py
        do_constant_folding (bool, default False): If True, the constant-folding
            optimization is applied to the model during export. Constant-folding
            optimization will replace some of the ops that have all constant
            inputs, with pre-computed constant nodes.
        example_outputs (tuple of Tensors, default None): example_outputs must be provided
            when exporting a ScriptModule or TorchScript Function.
        strip_doc_string (bool, default True): if True, strips the field
            "doc_string" from the exported model, which information about the stack
            trace.
        example_outputs: example outputs of the model that is being exported.
        dynamic_axes (dict<string, dict<int, string>> or dict<string, list(int)>, default empty dict):
            a dictionary to specify dynamic axes of input/output, such that:
            - KEY:  input and/or output names
            - VALUE: index of dynamic axes for given key and potentially the name to be used for
            exported dynamic axes. In general the value is defined according to one of the following
            ways or a combination of both:
            (1). A list of integers specifiying the dynamic axes of provided input. In this scenario
            automated names will be generated and applied to dynamic axes of provided input/output
            during export.
            OR (2). An inner dictionary that specifies a mapping FROM the index of dynamic axis in
            corresponding input/output TO the name that is desired to be applied on such axis of
            such input/output during export.
            Example. if we have the following shape for inputs and outputs:
                shape(input_1) = ('b', 3, 'w', 'h')
                and shape(input_2) = ('b', 4)
                and shape(output)  = ('b', 'd', 5)

            Then dynamic axes can be defined either as:
                (a). ONLY INDICES:
                    dynamic_axes = {'input_1':[0, 2, 3], 'input_2':[0], 'output':[0, 1]}

                    where automatic names will be generated for exported dynamic axes

                (b). INDICES WITH CORRESPONDING NAMES:
                    dynamic_axes = {'input_1':{0:'batch', 1:'width', 2:'height'},
                    'input_2':{0:'batch'},
                    'output':{0:'batch', 1:'detections'}

                    where provided names will be applied to exported dynamic axes

                (c). MIXED MODE OF (a) and (b)
                    dynamic_axes = {'input_1':[0, 2, 3], 'input_2':{0:'batch'}, 'output':[0,1]}
        keep_initializers_as_inputs (bool, default None): If True, all the initializers
            (typically corresponding to parameters) in the exported graph will also be
            added as inputs to the graph. If False, then initializers are not added as
            inputs to the graph, and only the non-parameter inputs are added as inputs.
            This may allow for better optimizations (such as constant folding etc.) by
            backends/runtimes that execute these graphs. If unspecified (default None),
            then the behavior is chosen automatically as follows. If operator_export_type
            is OperatorExportTypes.ONNX, the behavior is equivalent to setting this
            argument to False. For other values of operator_export_type, the behavior is
            equivalent to setting this argument to True.
    """

    from torch.onnx import utils
    return utils.export(model, args, f, export_params, verbose, training,
                        input_names, output_names, aten, export_raw_ir,
                        operator_export_type, opset_version, _retain_param_name,
                        do_constant_folding, example_outputs,
                        strip_doc_string, dynamic_axes, keep_initializers_as_inputs)


def export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils.export_to_pretty_string(*args, **kwargs)


def _export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils._export_to_pretty_string(*args, **kwargs)


def _optimize_trace(trace, operator_export_type):
    from torch.onnx import utils
    trace.set_graph(utils._optimize_graph(trace.graph(), operator_export_type))


def set_training(model, mode):
    r"""
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.
    """

    from torch.onnx import utils
    return utils.set_training(model, mode)


def _run_symbolic_function(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_function(*args, **kwargs)


def _run_symbolic_method(*args, **kwargs):
    from torch.onnx import utils
    return utils._run_symbolic_method(*args, **kwargs)


def is_in_onnx_export():
    r"""
    Check whether it's in the middle of the ONNX export.
    This function returns True in the middle of torch.onnx.export().
    torch.onnx.export should be executed with single thread.
    """

    from torch.onnx import utils
    return utils.is_in_onnx_export()


def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):
    from torch.onnx import utils
    return utils.register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version)
