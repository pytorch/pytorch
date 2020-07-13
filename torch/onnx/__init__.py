import torch._C as _C

TensorProtoDataType = _C._onnx.TensorProtoDataType
OperatorExportTypes = _C._onnx.OperatorExportTypes
TrainingMode = _C._onnx.TrainingMode
PYTORCH_ONNX_CAFFE2_BUNDLE = _C._onnx.PYTORCH_ONNX_CAFFE2_BUNDLE

ONNX_ARCHIVE_MODEL_PROTO_NAME = "__MODEL_PROTO"

# TODO: Update these variables when there
# is a new ir_version and producer_version
# and use these values in the exporter
ir_version = _C._onnx.IR_VERSION
producer_name = "pytorch"
producer_version = _C._onnx.PRODUCER_VERSION
constant_folding_opset_versions = [9, 10, 11, 12]


class ExportTypes:
    PROTOBUF_FILE = 1
    ZIP_ARCHIVE = 2
    COMPRESSED_ZIP_ARCHIVE = 3
    DIRECTORY = 4


def _export(*args, **kwargs):
    from torch.onnx import utils
    result = utils._export(*args, **kwargs)
    return result


def export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=None, _retain_param_name=True,
           do_constant_folding=True, example_outputs=None, strip_doc_string=True,
           dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
           enable_onnx_checker=True, use_external_data_format=False):
    r"""
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported;
    at the moment, it supports a limited set of dynamic models (e.g., RNNs.)

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (tuple of arguments or torch.Tensor): the inputs to
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
        training (enum, default TrainingMode.EVAL):
            TrainingMode.EVAL: export the model in inference mode.
            TrainingMode.PRESERVE: export the model in inference mode if model.training is
            False and to a training friendly mode if model.training is True.
            TrainingMode.TRAINING: export the model in a training friendly mode.
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
            OperatorExportTypes.ONNX: All ops are exported as regular ONNX ops
            (with ONNX namespace).
            OperatorExportTypes.ONNX_ATEN: All ops are exported as ATen ops
            (with aten namespace).
            OperatorExportTypes.ONNX_ATEN_FALLBACK: If an ATen op is not supported
            in ONNX or its symbolic is missing, fall back on ATen op. Registered ops
            are exported to ONNX regularly.
            Example graph::

                graph(%0 : Float)::
                  %3 : int = prim::Constant[value=0]()
                  %4 : Float = aten::triu(%0, %3) # missing op
                  %5 : Float = aten::mul(%4, %0) # registered op
                  return (%5)

            is exported as::

                graph(%0 : Float)::
                  %1 : Long() = onnx::Constant[value={0}]()
                  %2 : Float = aten::ATen[operator="triu"](%0, %1)  # missing op
                  %3 : Float = onnx::Mul(%2, %0) # registered op
                  return (%3)

            In the above example, aten::triu is not supported in ONNX, hence
            exporter falls back on this op.
            OperatorExportTypes.RAW: Export raw ir.
            OperatorExportTypes.ONNX_FALLTHROUGH: If an op is not supported
            in ONNX, fall through and export the operator as is, as a custom 
            ONNX op. Using this mode, the op can be exported and implemented by
            the user for their runtime backend.
            Example graph::

                graph(%x.1 : Long(1:1))::
                  %1 : None = prim::Constant()
                  %2 : Tensor = aten::sum(%x.1, %1)
                  %y.1 : Tensor[] = prim::ListConstruct(%2)
                  return (%y.1)

            is exported as::

                graph(%x.1 : Long(1:1))::
                  %1 : Tensor = onnx::ReduceSum[keepdims=0](%x.1)
                  %y.1 : Long() = prim::ListConstruct(%1)
                  return (%y.1)

            In the above example, prim::ListConstruct is not supported, hence
            exporter falls through.

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
        example_outputs (tuple of Tensors, default None): Model's example outputs being exported.
            example_outputs must be provided when exporting a ScriptModule or TorchScript Function.
        strip_doc_string (bool, default True): if True, strips the field
            "doc_string" from the exported model, which information about the stack
            trace.
        dynamic_axes (dict<string, dict<int, string>> or dict<string, list(int)>, default empty dict):
            a dictionary to specify dynamic axes of input/output, such that:
            - KEY:  input and/or output names
            - VALUE: index of dynamic axes for given key and potentially the name to be used for
            exported dynamic axes. In general the value is defined according to one of the following
            ways or a combination of both:
            (1). A list of integers specifying the dynamic axes of provided input. In this scenario
            automated names will be generated and applied to dynamic axes of provided input/output
            during export.
            OR (2). An inner dictionary that specifies a mapping FROM the index of dynamic axis in
            corresponding input/output TO the name that is desired to be applied on such axis of
            such input/output during export.

            Example. if we have the following shape for inputs and outputs:

            .. code-block:: none

                shape(input_1) = ('b', 3, 'w', 'h')
                and shape(input_2) = ('b', 4)
                and shape(output)  = ('b', 'd', 5)

            Then `dynamic axes` can be defined either as:

            1. ONLY INDICES::

                ``dynamic_axes = {'input_1':[0, 2, 3],
                                  'input_2':[0],
                                  'output':[0, 1]}``
                where automatic names will be generated for exported dynamic axes

            2. INDICES WITH CORRESPONDING NAMES::

                ``dynamic_axes = {'input_1':{0:'batch',
                                             1:'width',
                                             2:'height'},
                                  'input_2':{0:'batch'},
                                  'output':{0:'batch',
                                            1:'detections'}``
                where provided names will be applied to exported dynamic axes

            3. MIXED MODE OF (1) and (2)::

                ``dynamic_axes = {'input_1':[0, 2, 3],
                                  'input_2':{0:'batch'},
                                  'output':[0,1]}``

        keep_initializers_as_inputs (bool, default None): If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.

            This may allow for better optimizations (such as constant folding
            etc.) by backends/runtimes that execute these graphs. If
            unspecified (default None), then the behavior is chosen
            automatically as follows. If operator_export_type is
            OperatorExportTypes.ONNX, the behavior is equivalent to setting
            this argument to False. For other values of operator_export_type,
            the behavior is equivalent to setting this argument to True. Note
            that for ONNX opset version < 9, initializers MUST be part of graph
            inputs. Therefore, if opset_version argument is set to a 8 or
            lower, this argument will be ignored.
        custom_opsets (dict<string, int>, default empty dict): A dictionary to indicate
            custom opset domain and version at export. If model contains a custom opset,
            it is optional to specify the domain and opset version in the dictionary:
            - KEY: opset domain name
            - VALUE: opset version
            If the custom opset is not provided in this dictionary, opset version is set
            to 1 by default.
        enable_onnx_checker (bool, default True): If True the onnx model checker will be run
            as part of the export, to ensure the exported model is a valid ONNX model.
        external_data_format (bool, default False): If True, then the model is exported
            in ONNX external data format, in which case some of the model parameters are stored
            in external binary files and not in the ONNX model file itself. See link for format
            details: 
            https://github.com/onnx/onnx/blob/8b3f7e2e7a0f2aba0e629e23d89f07c7fc0e6a5e/onnx/onnx.proto#L423
            Also, in this case,  argument 'f' must be a string specifying the location of the model.
            The external binary files will be stored in the same location specified by the model 
            location 'f'. If False, then the model is stored in regular format, i.e. model and
            parameters are all in one file. This argument is ignored for all export types other
            than ONNX. 
    """

    from torch.onnx import utils
    return utils.export(model, args, f, export_params, verbose, training,
                        input_names, output_names, aten, export_raw_ir,
                        operator_export_type, opset_version, _retain_param_name,
                        do_constant_folding, example_outputs,
                        strip_doc_string, dynamic_axes, keep_initializers_as_inputs,
                        custom_opsets, enable_onnx_checker, use_external_data_format)


def export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils.export_to_pretty_string(*args, **kwargs)


def _export_to_pretty_string(*args, **kwargs):
    from torch.onnx import utils
    return utils._export_to_pretty_string(*args, **kwargs)


def _optimize_trace(graph, operator_export_type):
    from torch.onnx import utils
    return utils._optimize_graph(graph, operator_export_type)


def select_model_mode_for_export(model, mode):
    r"""
    A context manager to temporarily set the training mode of 'model'
    to 'mode', resetting it when we exit the with-block.  A no-op if
    mode is None.

    In version 1.6 changed to this from set_training
    """

    from torch.onnx import utils
    return utils.select_model_mode_for_export(model, mode)


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
