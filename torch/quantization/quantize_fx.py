from .fx import Fuser  # noqa: F401
from .fx import Quantizer  # noqa: F401
from torch.fx import GraphModule  # type: ignore
from .fx.utils import graph_pretty_str  # noqa: F401

def _check_is_graph_module(model):
    if not isinstance(model, GraphModule):
        raise ValueError(
            'input model must be a GraphModule, ' +
            'please run torch.fx.symbolic_trace on your model before using ' +
            'quantize_fx. Got type:' + str(type(model)))

def fuse_fx(graph_module, inplace=False):
    r""" Fuse modules in preparation for quantization

    Args:
        graph_module: GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
    _check_is_graph_module(graph_module)
    fuser = Fuser()
    return fuser.fuse(graph_module, inplace)

def _prepare_fx(graph_module, qconfig_dict, inplace, is_standalone_module=False):
    r""" Internal helper function for prepare_fx
    Args:
      `graph_modul`e, `qconfig_dict`, `inplace`: see docs for :func:`~torch.quantization.prepare_fx`
      `is_standalone_module`: a boolean flag indicates whether we are
      quantizing a standalone module or not, a standalone module
      is a submodule of the parent module that is not inlined in the
forward graph of the parent module,
      the way we quantize standalone module is described in:
      :func:`~torch.quantization._prepare_standalone_module_fx`
    """
    _check_is_graph_module(graph_module)
    graph_module = fuse_fx(graph_module, inplace)
    quantizer = Quantizer()
    return quantizer.prepare(graph_module, qconfig_dict, inplace=True, is_standalone_module=is_standalone_module)

def _prepare_standalone_module_fx(graph_module, qconfig_dict, inplace=False):
    r""" [Internal use only] Prepare a standalone module, so that it can be used when quantizing the
    parent module.
    standalone_module means it a submodule that is not inlined in parent module,
        and will be quantized separately as one unit.

    input of the module is quantized in parent module, output of the module
    is quantized in the standalone module.
    Extra attributes in output GraphModule while preparing a standalone module:
        _standalone_module_observed_input_idxs(List[Int]): a list of indexs for the graph inputs that
                                         needs to be observed in parent module
        _output_is_observed(Bool): a boolean variable indicate whether the output of the
                                   custom module is observed or not

    """
    return _prepare_fx(graph_module, qconfig_dict, inplace, is_standalone_module=True)

def prepare_fx(graph_module, qconfig_dict, inplace=False):
    r""" Prepare a model for post training static quantization

    Args:
      `graph_module`: model from symbolic_tracing (torch.fx.symbolic_trace), must be
      an eval model
      `qconfig_dict`: qconfig_dict is a dictionary with the following configurations:
      qconfig_dict = {
      # optional, global config
      "": qconfig?,

      # optional, used for module and function types
      # could also be split into module_types and function_types if we prefer
      "object_type": [
        (torch.nn.Conv2d, qconfig?),
        (torch.nn.functional.add, qconfig?),
        ...,
       ],

      # optional, used for module names
      "module_name": [
        ("foo.bar", qconfig?)
        ...,
      ],

      # optional, matched in order, first match takes precedence
      "module_name_regex": [
        ("foo.*bar.*conv[0-9]+", qconfig?)
        ...,
      ],
      # priority (in increasing order): global, object_type, module_name_regex, module_name
      # qconfig == None means fusion and quantization should be skipped for anything
      # matching the rule

      # optional: specify the path for standalone modules
      # These modules are symbolically traced and quantized as one unit
      # User should also skip symbolic tracing through these modules
      # so that the call to the submodule appears as one call_module
      # node in the forward graph of the GraphModule
      "standalone_module_name": [
         "submodule.standalone"
      ]
      }
      `inplace`: flag for carry out model transformations in-place,
      the original module is mutated


    Return:
      A GraphModule with observer (configured by qconfig_dict), ready for calibration

    Example:
    ```python
    import torch
    from torch.quantization import get_default_qconfig
    from torch.quantization import prepare_fx

    float_model.eval()
    graph_module = torch.fx.symbolic_trace(float_model)
    qconfig = get_default_qconfig('fbgemm')
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for image, target in data_loader:
                model(image)

    qconfig_dict = {"": qconfig}
    prepared_model = prepare_fx(graph_module, qconfig_dict)
    # Run calibration
    calibrate(prepared_model, sample_inference_data)
    """
    assert not graph_module.training, 'prepare_fx only works for models in' + \
        'eval mode'
    return _prepare_fx(graph_module, qconfig_dict, inplace)

def prepare_static_fx(graph_module, qconfig_dict, inplace=False):
    assert not graph_module.training, 'prepare_static_fx only works for models in ' + \
        'eval mode'
    return prepare_fx(graph_module, qconfig_dict, inplace)

def prepare_qat_fx(graph_module, qconfig_dict, inplace=False):
    r""" Prepare a model for quantization aware training
    Args:
      `graph_module`: model from symbolic_tracing (torch.fx.symbolic_trace), must be
       a train model
      `qconfig_dict`: see :func:`~torch.quantization.prepare_fx`
      `inplace`: flag for carry out model transformations in-place,
       the original module is mutated

    Return:
      A GraphModule with fake quant modules (configured by qconfig_dict), ready for
      quantization aware training

    Example:
    ```python
    import torch
    from torch.quantization import get_default_qat_qconfig
    from torch.quantization import prepare_fx

    float_model.train()
    graph_module = torch.fx.symbolic_trace(float_model)
    qconfig = get_default_qat_qconfig('fbgemm')
    def train_loop(model, train_data):
        model.train()
        for image, target in data_loader:
            ...

    qconfig_dict = {"": qconfig}
    prepared_model = prepare_fx(graph_module, qconfig_dict)
    # Run calibration
    train_loop(prepared_model, train_loop)
    """
    assert graph_module.training, 'prepare_qat_fx only works for models in ' + \
        'train mode'
    return _prepare_fx(graph_module, qconfig_dict, inplace)

def _convert_fx(graph_module, inplace, debug, is_standalone_module=False):
    """ `is_standalone_module`: see docs in :func:`~torch.quantization.prepare_standalone_module_fx`
    """
    _check_is_graph_module(graph_module)
    quantizer = Quantizer()
    return quantizer.convert(graph_module, inplace, debug, is_standalone_module)

def convert_fx(graph_module, inplace=False, debug=False):
    r""" Convert a calibrated or trained model to a quantized model
    Args:
        `graph_module`: A prepared and calibrated/trained model (GraphModule)
        `inplace`: flag for carry out model transformations in-place,
        the original module is mutated
        `debug`: flag for producing a debug friendly model (preserve weight attribute)
    Return:
        A quantized model (GraphModule)

    Example:
    ```python
    # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
    quantized_model = convert_fx(prepared_model)
    ```
    """
    return _convert_fx(graph_module, inplace, debug)

def _convert_standalone_module_fx(graph_module, inplace=False, debug=False):
    r""" [Internal use only] Convert a model produced by :func:`~torch.quantization.prepare_standalone_module_fx`
    and convert it to a quantized model

    The inputs will be quantized by parent module, checks `_standalone_module_observed_input_idxs` of
    input model and will treat these inputs as quantized
    also will not dequantize the final output
    Return:
      A quantized standalone module which accepts quantized input(if needed)
      and produces quantized output (if needed).
    """
    return _convert_fx(graph_module, inplace, debug, is_standalone_module=True)
