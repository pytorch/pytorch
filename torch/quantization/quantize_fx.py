from .fx import Fuser  # noqa: F401
from .fx import Quantizer  # noqa: F401

def fuse_fx(graph_module, inplace=False):
    fuser = Fuser()
    return fuser.fuse(graph_module, inplace)

def _prepare_fx(graph_module, qconfig_dict, inplace, is_dynamic):
    quantizer = Quantizer()
    prepare = quantizer.prepare_dynamic if is_dynamic else quantizer.prepare
    prepared = prepare(graph_module, qconfig_dict, inplace)
    return prepared

def prepare_fx(graph_module, qconfig_dict, inplace=False):
    """ If graph_module is in training mode, the model will be
    prepared as a qat model, otherwise, it will be prepared as a
    model used for post training static quantization.

    Args:
      graph_module: model from symbolic_tracing (torch.fx)
      qconfig_dict: tbd
    """
    return _prepare_fx(graph_module, qconfig_dict, inplace, False)

def prepare_dynamic_fx(graph_module, qconfig_dict, inplace=False):
    return _prepare_fx(graph_module, qconfig_dict, inplace, True)

def _convert_fx(graph_module, inplace=False, debug=False, is_dynamic=False):
    quantizer = Quantizer()
    return quantizer.convert(graph_module, inplace, debug, is_dynamic)

def convert_fx(graph_module, inplace=False, debug=False):
    return _convert_fx(graph_module, inplace, debug, is_dynamic=False)

def convert_dynamic_fx(graph_module, inplace=False, debug=False):
    return _convert_fx(graph_module, inplace, debug, is_dynamic=True)

def _quantize_fx(model, qconfig_dict, run_fn=None, run_args=None, inplace=False, debug=False, is_dynamic=False):
    if is_dynamic:
        model = prepare_dynamic_fx(model, qconfig_dict, inplace)
        # TODO: change inplace to True since the model is already copied in
        # prepare
        model = convert_dynamic_fx(model, False, debug)
    else:
        assert run_fn, "Must provide calibration function for post training static quantization"
        assert run_args, "Must provide calibration dataset for post training static quantization"
        model = prepare_fx(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        # TODO: change inplace to True since the model is already copied in
        # prepare
        model = convert_fx(model, False, debug)

    return model


def quantize_fx(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    r"""Quantize the input float symbolically traced GraphModule model with
    post training static quantization (if the model is in eval mode)
    or quantization aware training (if the model is in training mode).

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, empty key means the qconfig will be applied
        to whole model unless it’s overwritten by more specific configurations, the
        qconfig for each module is either found in the dictionary or fallback to
         the qconfig of parent module.

        Right now qconfig_dict is the only way to configure how the model is quantized,
        and it is done in the granularity of module, that is, we only support one type
        of qconfig for each torch.nn.Module, and the qconfig for sub module will
        override the qconfig for parent module, empty string means global configuration.
        `run_fn`: a calibration function for calibrating the prepared model
        `run_args`: positional arguments for `run_fn`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.quantization import get_default_qconfig
    from torch.quantization import quantize_fx

    graph_module = torch.fx.symbolic_trace(float_model.eval())
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    quantized_model = quantize_fx(
        graph_module,
        {'': qconfig},
        calibrate,
        [data_loader_test])
    ```
    """
    return _quantize_fx(
        model, qconfig_dict, run_fn, run_args, inplace, debug, is_dynamic=Falsse)

def quantize_dynamic_fx(model, qconfig_dict, inplace=False, debug=False):
    r"""Quantize the input float symbolically traced GraphModule model with
    post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        descriptions in :func:`~torch.quantization.quantize_fx`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.quantization import per_channel_dynamic_qconfig
    from torch.quantization import quantize_dynmiac_fx

    graph_module = torch.fx.symbolic_trace(float_model.eval())
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    quantized_model = quantize_dynamic_fx(
        graph_module,
        {'': qconfig},
        calibrate,
        [data_loader_test])
    ```
    """
    return _quantize_fx(
        model, qconfig_dict, inplace=inplace, debug=debug, is_dynamic=True)
