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
constant_folding_opset_versions = [9, 10, 11, 12, 13, 14]


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
           input_names=None, output_names=None, operator_export_type=None,
           opset_version=None, _retain_param_name=None, do_constant_folding=True,
           example_outputs=None, strip_doc_string=None, dynamic_axes=None,
           keep_initializers_as_inputs=None, custom_opsets=None, enable_onnx_checker=None,
           use_external_data_format=None):
    r"""
    Exports a model into ONNX format. If ``model`` is not a
    :class:`torch.jit.ScriptModule` nor a :class:`torch.jit.ScriptFunction`, this runs
    ``model`` once in order to convert it to a TorchScript graph to be exported
    (the equivalent of :func:`torch.jit.trace`). Thus this has the same limited support
    for dynamic control flow as :func:`torch.jit.trace`.

    Args:
        model (torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction):
            the model to be exported.
        args (tuple or torch.Tensor):

            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS::

                args = (x, y, z)

            The tuple should contain model inputs such that ``model(*args)`` is a valid
            invocation of the model. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.

            2. A TENSOR::

                args = torch.Tensor([1])

            This is equivalent to a 1-ary tuple of that Tensor.

            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED ARGUMENTS::

                args = (x,
                        {'y': input_y,
                         'z': input_z})

            All but the last element of the tuple will be passed as non-keyword arguments,
            and named arguments will be set from the last element. If a named argument is
            not present in the dictionary, it is assigned the default value, or None if a
            default value is not provided.

            .. note::
                If a dictionary is the last element of the args tuple, it will be
                interpreted as containing named arguments. In order to pass a dict as the
                last non-keyword arg, provide an empty dict as the last element of the args
                tuple. For example, instead of::

                    torch.onnx.export(
                        model,
                        (x,
                         # WRONG: will be interpreted as named arguments
                         {y: z}),
                        "test.onnx.pb")

                Write::

                    torch.onnx.export(
                        model,
                        (x,
                         {y: z},
                         {}),
                        "test.onnx.pb")

        f: a file-like object (such that ``f.fileno()`` returns a file descriptor)
            or a string containing a file name.  A binary protocol buffer will be written
            to this file.
        export_params (bool, default True): if True, all parameters will
            be exported. Set this to False if you want to export an untrained model.
            In this case, the exported model will first take all of its parameters
            as arguments, with the ordering as specified by ``model.state_dict().values()``
        verbose (bool, default False): if True, prints a description of the
            model being exported to stdout. In addition, the final ONNX graph will include the
            field ``doc_string``` from the exported model which mentions the source code locations
            for ``model``.
        training (enum, default TrainingMode.EVAL):
            * ``TrainingMode.EVAL``: export the model in inference mode.
            * ``TrainingMode.PRESERVE``: export the model in inference mode if model.training is
              False and in training mode if model.training is True.
            * ``TrainingMode.TRAINING``: export the model in training mode. Disables optimizations
              which might interfere with training.
        input_names (list of str, default empty list): names to assign to the
            input nodes of the graph, in order.
        output_names (list of str, default empty list): names to assign to the
            output nodes of the graph, in order.
        operator_export_type (enum, default None):

            None usually means ``OperatorExportTypes.ONNX``.
            However if PyTorch was built with ``-DPYTORCH_ONNX_CAFFE2_BUNDLE``, None means
            ``OperatorExportTypes.ONNX_ATEN_FALLBACK``.

            * ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops
              (in the default opset domain).
            * ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops
              to standard ONNX ops in the default opset domain. If unable to do so
              (e.g. because support has not been added to convert a particular torch op to ONNX),
              fall back to exporting the op into a custom opset domain without conversion. Applies
              to `custom ops <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_
              as well as ATen ops. For the exported model to be usable, the runtime must support
              these non-standard ops.
            * ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the TorchScript namespace "aten")
              are exported as ATen ops (in opset domain "org.pytorch.aten").
              `ATen <https://pytorch.org/cppdocs/#aten>`_ is PyTorch's built-in tensor library, so
              this instructs the runtime to use PyTorch's implementation of these ops.

              .. warning::

                Models exported this way are probably runnable only by Caffe2.

              This may be useful if the numeric differences in implementations of operators are
              causing large differences in behavior between PyTorch and Caffe2 (which is more
              common on untrained models).

            * ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each ATen op
              (in the TorchScript namespace "aten") as a regular ONNX op. If we are unable to do so
              (e.g. because support has not been added to convert a particular torch op to ONNX),
              fall back to exporting an ATen op. See documentation on OperatorExportTypes.ONNX_ATEN for
              context.
              For example::

                graph(%0 : Float):
                  %3 : int = prim::Constant[value=0]()
                  # conversion unsupported
                  %4 : Float = aten::triu(%0, %3)
                  # conversion supported
                  %5 : Float = aten::mul(%4, %0)
                  return (%5)

              Assuming ``aten::triu`` is not supported in ONNX, this will be exported as::

                graph(%0 : Float):
                  %1 : Long() = onnx::Constant[value={0}]()
                  # not converted
                  %2 : Float = aten::ATen[operator="triu"](%0, %1)
                  # converted
                  %3 : Float = onnx::Mul(%2, %0)
                  return (%3)

              If an op is in the TorchScript namespace "quantized", it will be exported
              in the ONNX opset domain "caffe2". These ops are produced by
              the modules described in
              `Quantization <https://pytorch.org/docs/stable/quantization.html>`_.

              .. warning::

                Models exported this way are probably runnable only by Caffe2.

        opset_version (int, default 9):
            Must be ``== _onnx_main_opset or in _onnx_stable_opsets``,
            defined in torch/onnx/symbolic_helper.py.
        _retain_param_name (bool, default True): [Deprecated and ignored. Will be removed in next PyTorch
            release]
        do_constant_folding (bool, default False): Apply the constant-folding optimization.
            Constant-folding will replace some of the ops that have all constant inputs
            with pre-computed constant nodes.
        example_outputs (T or a tuple of T, where T is Tensor or convertible to Tensor, default None):
            [Deprecated and ignored. Will be removed in next PyTorch release],
            Must be provided when exporting a ScriptModule or ScriptFunction, ignored otherwise.
            Used to determine the type and shape of the outputs without tracing the execution of
            the model. A single object is treated as equivalent to a tuple of one element.
        strip_doc_string (bool, default True): [Deprecated and ignored. Will be removed in next PyTorch release]
        dynamic_axes (dict<string, dict<int, string>> or dict<string, list(int)>, default empty dict):

            By default the exported model will have the shapes of all input and output tensors
            set to exactly match those given in ``args`` (and ``example_outputs`` when that arg is
            required). To specify axes of tensors as dynamic (i.e. known only at run-time), set
            ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be provided in ``input_names`` or
              ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
              list, each element is an axis index.

            For example::

                class SumModule(torch.nn.Module):
                    def forward(self, x):
                        return torch.sum(x, dim=1)

                torch.onnx.export(SumModule(), (torch.ones(2, 2),), "onnx.pb",
                                  input_names=["x"], output_names=["sum"])

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                ...

            While::

                torch.onnx.export(SumModule(), (torch.ones(2, 2),), "onnx.pb",
                                  input_names=["x"], output_names=["sum"],
                                  dynamic_axes={
                                      # dict value: manually named axes
                                      "x": {0: "my_custom_axis_name"},
                                      # list value: automatic names
                                      "sum": [0],
                                  })

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_param: "my_custom_axis_name"  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_param: "sum_dynamic_axes_1"  # axis 0
                ...

        keep_initializers_as_inputs (bool, default None): If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by
            backends/runtimes.

            If ``opset_version`` < 9, initializers MUST be part of graph
            inputs and this argument will be ignored and the behavior will be
            equivalent to setting this argument to True.

            If None, then the behavior is chosen automatically as follows:

            * If ``operator_export_type=OperatorExportTypes.ONNX``, the behavior is equivalent
              to setting this argument to False.
            * Else, the behavior is equivalent to setting this argument to True.

        custom_opsets (dict<str, int>, default empty dict): A dictionary to indicate

            A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1.

        enable_onnx_checker (bool, default True): Deprecated and ignored. Will be removed in next
            Pytorch release.
        use_external_data_format (bool, default False): [Deprecated and ignored. Will be removed in
            next Pytorch release.]
            If True, then some of the model parameters are stored in external data files and not in
            the ONNX model file itself. Models larger than 2GB cannot be exported in one file because
            of size limits imposed by Protocol Buffers.
            For details see
            `onnx.proto <https://github.com/onnx/onnx/blob/32c7cb66/onnx/onnx.proto#L562>`_.
            If True,  argument ``f`` must be a string specifying the location of the model.
            The external data files will be stored in the same directory as ``f``.
            This argument is ignored unless ``operator_export_type=OperatorExportTypes.ONNX``.

    Raises:
      ONNXCheckerError: If the ONNX checker detects an invalid ONNX graph. Will still export the
        model to the file ``f`` even if this is raised.
    """

    from torch.onnx import utils
    return utils.export(model, args, f, export_params, verbose, training,
                        input_names, output_names, operator_export_type, opset_version,
                        _retain_param_name, do_constant_folding, example_outputs,
                        strip_doc_string, dynamic_axes, keep_initializers_as_inputs,
                        custom_opsets, enable_onnx_checker, use_external_data_format)


def export_to_pretty_string(*args, **kwargs) -> str:
    r"""
    Similar to :func:`export`, but returns a text representation of the ONNX
    model. Only differences in args listed below. All other args are the same
    as :func:`export`.

    Args:
      f:  Deprecated and ignored. Will be removed in the next release of
          PyTorch.
      add_node_names (bool, default True): Whether or not to set
          NodeProto.name. This makes no difference unless
          ``google_printer=True``.
      google_printer (bool, default False): If False, will return a custom,
          compact representation of the model. If True will return the
          protobuf's `Message::DebugString()`, which is more verbose.

    Returns:
      A UTF-8 str containing a human-readable representation of the ONNX model.
    """
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
    A context manager to temporarily set the training mode of ``model``
    to ``mode``, resetting it when we exit the with-block.  A no-op if
    mode is None.

    Args:
      model: Same type and meaning as ``model`` arg to :func:`export`.
      mode: Same type and meaning as ``training`` arg to :func:`export`.
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
    Returns True iff :func:`export` is running in the current thread
    """

    from torch.onnx import utils
    return utils.is_in_onnx_export()


def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):
    r"""
    Registers ``symbolic_fn`` to handle ``symbolic_name``. See
    "Custom Operators" in the module documentation for an example usage.

    Args:
      symbolic_name (str): The name of the custom operator in "<domain>::<op>"
        format.
      symbolic_fn (Callable): A function that takes in the ONNX graph and
        the input arguments to the current operator, and returns new
        operator nodes to add to the graph.
      opset_version (int): The ONNX opset version in which to register.
    """
    from torch.onnx import utils
    utils.register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version)


def unregister_custom_op_symbolic(symbolic_name, opset_version):
    from torch.onnx import utils
    utils.unregister_custom_op_symbolic(symbolic_name, opset_version)
