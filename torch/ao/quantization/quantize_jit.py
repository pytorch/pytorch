
import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module

def _check_is_script_module(model):
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError('input must be a script module, got: ' + str(type(model)))

def _check_forward_method(model):
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')

def script_qconfig(qconfig):
    r"""Instantiate the activation and weight observer modules and script
    them, these observer module instances will be deepcopied during
    prepare_jit step.
    """
    return QConfig(
        activation=torch.jit.script(qconfig.activation())._c,
        weight=torch.jit.script(qconfig.weight())._c)

def script_qconfig_dict(qconfig_dict):
    r"""Helper function used by `prepare_jit`.
    Apply `script_qconfig` for all entries in `qconfig_dict` that is
    not None.
    """
    return {k: script_qconfig(v) if v else None for k, v in qconfig_dict.items()}

def fuse_conv_bn_jit(model, inplace=False):
    r""" Fuse conv - bn module
    Works for eval model only.

    Args:
        model: TorchScript model from scripting or tracing
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.fuse_conv_bn_jit")
    model_c = model._c
    model_c = torch._C._jit_pass_fold_convbn(model_c)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

def _prepare_jit(model, qconfig_dict, inplace=False, quant_type=QuantType.STATIC):
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all(isinstance(x, str) for x in qconfig_dict.keys()):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observers(model._c,
                                                  'forward',
                                                  scripted_qconfig_dict,
                                                  inplace,
                                                  quant_type)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

def prepare_jit(model, qconfig_dict, inplace=False):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.STATIC)

def prepare_dynamic_jit(model, qconfig_dict, inplace=False):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.prepare_dynamic_jit")
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)

def _convert_jit(model, inplace=False, debug=False, quant_type=QuantType.STATIC,
                 preserved_attrs=None):
    _check_is_script_module(model)
    model.eval()
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant(model_c, 'forward', inplace, debug, quant_type)
    if not debug:
        is_xpu = all(p.device.type == 'xpu' for p in model.parameters())
        if not is_xpu:
            # Moving model parameters to CPU since quantized operators
            # are only supported on CPU and XPU right now
            model.cpu()
        if preserved_attrs is None:
            preserved_attrs = []
        model_c = torch._C._jit_pass_quant_finalize(model_c, quant_type, preserved_attrs)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model

def convert_jit(model, inplace=False, debug=False, preserved_attrs=None):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_jit")
    return _convert_jit(model, inplace, debug, quant_type=QuantType.STATIC, preserved_attrs=preserved_attrs)

def convert_dynamic_jit(model, inplace=False, debug=False, preserved_attrs=None):
    torch._C._log_api_usage_once("quantization_api.quantize_jit.convert_dynamic_jit")
    return _convert_jit(model, inplace, debug, quant_type=QuantType.DYNAMIC, preserved_attrs=preserved_attrs)

def _quantize_jit(model, qconfig_dict, run_fn=None, run_args=None, inplace=False, debug=False, quant_type=QuantType.STATIC):
    # Always do inplace convert because the Tensor is already
    # copied in prepare_jit when inplace is False
    if quant_type == QuantType.DYNAMIC:
        model = prepare_dynamic_jit(model, qconfig_dict, inplace)
        model = convert_dynamic_jit(model, True, debug)
    else:
        assert run_fn, "Must provide calibration function for post training static quantization"
        assert run_args, "Must provide calibration dataset for post training static quantization"
        model = prepare_jit(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        model = convert_jit(model, True, debug)

    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model

def quantize_jit(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    r"""Quantize the input float TorchScript model with
    post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, empty key means the qconfig will be applied
        to whole model unless it's overwritten by more specific configurations, the
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
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization import quantize_jit

    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    quantized_model = quantize_jit(
        ts_model,
        {'': qconfig},
        calibrate,
        [data_loader_test])
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.quantize_jit")
    return _quantize_jit(model, qconfig_dict, run_fn, run_args, inplace, debug, quant_type=QuantType.STATIC)

def quantize_dynamic_jit(model, qconfig_dict, inplace=False, debug=False):
    r"""Quantize the input float TorchScript model with
    post training dynamic quantization.
    Currently only qint8 quantization of torch.nn.Linear is supported.

    Args:
        `model`: input float TorchScript model
        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and
        qconfig for that module as value, please see detailed
        descriptions in :func:`~torch.ao.quantization.quantize_jit`
        `inplace`: carry out model transformations in-place, the original module is
        mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)

    Return:
        Quantized TorchSciprt model.

    Example:
    ```python
    import torch
    from torch.ao.quantization import per_channel_dynamic_qconfig
    from torch.ao.quantization import quantize_dynmiac_jit

    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    quantized_model = quantize_dynamic_jit(
        ts_model,
        {'': qconfig},
        calibrate,
        [data_loader_test])
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_jit.quantize_dynamic_jit")
    return _quantize_jit(model, qconfig_dict, inplace=inplace, debug=debug, quant_type=QuantType.DYNAMIC)
