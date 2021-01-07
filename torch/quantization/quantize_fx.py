import torch
from torch.fx import GraphModule  # type: ignore
from torch.fx.symbolic_trace import Tracer  # type: ignore
from .fx import Fuser  # noqa: F401
from .fx import Quantizer  # noqa: F401
from .fx.utils import graph_pretty_str  # noqa: F401
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from torch.nn.intrinsic import _FusedModule
from typing import Dict, Any, List, Callable

def _check_is_graph_module(model: torch.nn.Module) -> None:
    if not isinstance(model, GraphModule):
        raise ValueError(
            'input model must be a GraphModule, ' +
            'Got type:' + str(type(model)) + ' Please make ' +
            'sure to follow the tutorials.')

def _swap_ff_with_fxff(model: torch.nn.Module) -> None:
    r""" Swap FloatFunctional with FXFloatFunctional
    """
    modules_to_swap = []
    for name, module in model.named_children():
        if isinstance(module, torch.nn.quantized.FloatFunctional):
            modules_to_swap.append(name)
        else:
            _swap_ff_with_fxff(module)

    for name in modules_to_swap:
        del model._modules[name]
        model._modules[name] = torch.nn.quantized.FXFloatFunctional()

def _fuse_fx(
        graph_module: GraphModule,
        fuse_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" Internal helper function to fuse modules in preparation for quantization

    Args:
        graph_module: GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
    _check_is_graph_module(graph_module)
    fuser = Fuser()
    return fuser.fuse(graph_module, fuse_custom_config_dict)

class CustomTracer(Tracer):
    def __init__(self, skipped_module_names: List[str],
                 skipped_module_classes: List[Callable]):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes

    def is_leaf_module(self, m, module_qualified_name):
        return (m.__module__.startswith('torch.nn') and
                not isinstance(m, torch.nn.Sequential)) or \
            module_qualified_name in self.skipped_module_names or \
            type(m) in self.skipped_module_classes or \
            isinstance(m, _FusedModule)


def _prepare_fx(model: torch.nn.Module, qconfig_dict: Any,
                prepare_custom_config_dict: Dict[str, Any] = None,
                is_standalone_module: bool = False) -> GraphModule:
    r""" Internal helper function for prepare_fx
    Args:
      `model`, `qconfig_dict`, `prepare_custom_config_dict`: see docs for :func:`~torch.quantization.prepare_fx`
      `is_standalone_module`: a boolean flag indicates whether we are
      quantizing a standalone module or not, a standalone module
      is a submodule of the parent module that is not inlined in the
forward graph of the parent module,
      the way we quantize standalone module is described in:
      :func:`~torch.quantization._prepare_standalone_module_fx`
    """
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}

    skipped_module_names = prepare_custom_config_dict.get("non_traceable_module_name", [])
    skipped_module_classes = prepare_custom_config_dict.get("non_traceable_module_class", [])

    # swap FloatFunctional with FXFloatFunctional
    _swap_ff_with_fxff(model)

    # symbolically trace the model
    if not is_standalone_module:
        # standalone module and custom module config are applied in top level module
        standalone_module_name_configs = prepare_custom_config_dict.get("standalone_module_name", [])
        skipped_module_names += [config[0] for config in standalone_module_name_configs]

        standalone_module_class_configs = prepare_custom_config_dict.get("standalone_module_class", [])
        skipped_module_classes += [config[0] for config in standalone_module_class_configs]
        float_custom_module_classes = get_custom_module_class_keys(
            prepare_custom_config_dict, "float_to_observed_custom_module_class")
        skipped_module_classes += float_custom_module_classes
    tracer = CustomTracer(skipped_module_names, skipped_module_classes)
    graph_module = GraphModule(model, tracer.trace(model))
    graph_module = _fuse_fx(graph_module, prepare_custom_config_dict)
    quantizer = Quantizer()
    return quantizer.prepare(
        graph_module,
        qconfig_dict,
        prepare_custom_config_dict=prepare_custom_config_dict,
        is_standalone_module=is_standalone_module)

def _prepare_standalone_module_fx(
        model: torch.nn.Module, qconfig_dict: Any,
        prepare_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" [Internal use only] Prepare a standalone module, so that it can be used when quantizing the
    parent module.
    standalone_module means it a submodule that is not inlined in parent module,
        and will be quantized separately as one unit.

    Both input and output of the module are observed in the
    standalone module.
    """
    return _prepare_fx(model, qconfig_dict, prepare_custom_config_dict, is_standalone_module=True)

def fuse_fx(model: torch.nn.Module,
            fuse_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" Fuse modules like conv+bn, conv+bn+relu etc, model must be in eval mode.
    Fusion rules are defined in torch.quantization.fx.fusion_pattern.py
    Args:
        `model`: a torch.nn.Module model
        `fuse_custom_config_dict`: Dictionary for custom configurations for fuse_fx, e.g.
         fuse_custom_config_dict = {
           "additional_fuser_method_mapping": {
             (Module1, Module2): fuse_module1_module2
           }
         }

    Example:
    ```python
    from torch.quantization import fuse_fx
    m = Model().eval()
    m = fuse_fx(m)
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.fuse_fx")
    assert not model.training, 'fuse_fx only works on models in eval mode'
    graph_module = torch.fx.symbolic_trace(model)  # type: ignore
    return _fuse_fx(graph_module, fuse_custom_config_dict)

def prepare_fx(
        model: torch.nn.Module, qconfig_dict: Any,
        prepare_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" Prepare a model for post training static quantization

    Args:
      `model`: torch.nn.Module model, must be in eval mode
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
      }
      `prepare_custom_config_dict`: customization configuration dictionary for
      quantization tool:
      prepare_custom_config_dict = {
        # optional: specify the path for standalone modules
        # These modules are symbolically traced and quantized as one unit
        "standalone_module_name": [
           # module_name, qconfig_dict, prepare_custom_config_dict
           ("submodule.standalone",
            None,  # qconfig_dict for the prepare function called in the submodule,
                   # None means use qconfig from parent qconfig_dict
            {"input_quantized_idxs": [], "output_quantized_idxs": []})  # prepare_custom_config_dict
        ],

        "standalone_module_class": [
            # module_class, qconfig_dict, prepare_custom_config_dict
            (StandaloneModule,
             None,  # qconfig_dict for the prepare function called in the submodule,
                    # None means use qconfig from parent qconfig_dict
            {"input_quantized_idxs": [0], "output_quantized_idxs": [0]})  # prepare_custom_config_dict
        ],

        # user will manually define the corresponding observed
        # module class which has a from_float class method that converts
        # float custom module to observed custom module
        # (only needed for static quantization)
        "float_to_observed_custom_module_class": {
           "static": {
               CustomModule: ObservedCustomModule
           }
        },

        # the qualified names for the submodule that are not symbolically traceable
        "non_traceable_module_name": [
           "non_traceable_module"
        ],

        # the module classes that are not symbolically traceable
        # we'll also put dynamic/weight_only custom module here
        "non_traceable_module_class": [
           NonTraceableModule
        ],

        # Additional fuser_method mapping
        "additional_fuser_method_mapping": {
           (torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn
        },

        # Additioanl module mapping for qat
        "additional_qat_module_mapping": {
           torch.nn.intrinsic.ConvBn2d: torch.nn.qat.ConvBn2d
        },

        # Additional fusion patterns
        "additional_fusion_pattern": {
           (torch.nn.BatchNorm2d, torch.nn.Conv2d): ConvReluFusionhandler
        },

        # Additional quantization patterns
        "additional_quant_pattern": {
           torch.nn.Conv2d: ConvReluQuantizeHandler,
           (torch.nn.ReLU, torch.nn.Conv2d): ConvReluQuantizeHandler,
        }

        # By default, inputs and outputs of the graph are assumed to be in
        # fp32. Providing `input_quantized_idxs` will set the inputs with the
        # corresponding indices to be quantized. Providing
        # `output_quantized_idxs` will set the outputs with the corresponding
        # indices to be quantized.
        "input_quantized_idxs": [0],
        "output_quantized_idxs": [0],
      }


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
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_fx")
    assert not model.training, 'prepare_fx only works for models in ' + \
        'eval mode'
    return _prepare_fx(model, qconfig_dict, prepare_custom_config_dict)

def prepare_qat_fx(
        model: torch.nn.Module, qconfig_dict: Any,
        prepare_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" Prepare a model for quantization aware training
    Args:
      `model`: torch.nn.Module model, must be in train mode
      `qconfig_dict`: see :func:`~torch.quantization.prepare_fx`
      `prepare_custom_config_dict`: see :func:`~torch.quantization.prepare_fx`

    Return:
      A GraphModule with fake quant modules (configured by qconfig_dict), ready for
      quantization aware training

    Example:
    ```python
    import torch
    from torch.quantization import get_default_qat_qconfig
    from torch.quantization import prepare_fx

    qconfig = get_default_qat_qconfig('fbgemm')
    def train_loop(model, train_data):
        model.train()
        for image, target in data_loader:
            ...

    float_model.train()
    qconfig_dict = {"": qconfig}
    prepared_model = prepare_fx(float_model, qconfig_dict)
    # Run calibration
    train_loop(prepared_model, train_loop)
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_qat_fx")
    assert model.training, 'prepare_qat_fx only works for models in  ' + \
        'train mode'
    return _prepare_fx(model, qconfig_dict, prepare_custom_config_dict)

def _convert_fx(
        graph_module: GraphModule, debug: bool,
        convert_custom_config_dict: Dict[str, Any] = None,
        is_standalone_module: bool = False) -> GraphModule:
    """ `is_standalone_module`: see docs in :func:`~torch.quantization.prepare_standalone_module_fx`
    """
    _check_is_graph_module(graph_module)
    quantizer = Quantizer()
    return quantizer.convert(graph_module, debug, convert_custom_config_dict, is_standalone_module)

def convert_fx(
        graph_module: GraphModule, debug: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" Convert a calibrated or trained model to a quantized model
    Args:
        `graph_module`: A prepared and calibrated/trained model (GraphModule)
        `debug`: flag for producing a debug friendly model (preserve weight attribute)
        `convert_custom_config_dict`: dictionary for custom configurations for convert function:
        convert_custom_config_dict = {

          # addtional object (module/operator) mappings that will overwrite the default
          # module mappingn
          "additional_object_mapping": {
             "static": {
                FloatModule: QuantizedModule,
                float_op: quantized_op
             },
             "dynamic": {
                FloatModule: DynamicallyQuantizedModule,
                float_op: dynamically_quantized_op
             },
          }

          # user will manually define the corresponding quantized
          # module class which has a from_observed class method that converts
          # observed custom module to quantized custom module
          "observed_to_quantized_custom_module_class": {
             "static": {
                 ObservedCustomModule: QuantizedCustomModule
             },
             "dynamic": {
                 ObservedCustomModule: QuantizedCustomModule
             },
             "weight_only": {
                 ObservedCustomModule: QuantizedCustomModule
             }
          }
        }

    Return:
        A quantized model (GraphModule)

    Example:
    ```python
    # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
    quantized_model = convert_fx(prepared_model)
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_fx")
    return _convert_fx(graph_module, debug, convert_custom_config_dict)

def _convert_standalone_module_fx(
        graph_module: GraphModule, debug: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" [Internal use only] Convert a model produced by :func:`~torch.quantization.prepare_standalone_module_fx`
    and convert it to a quantized model

    Return:
        A quantized standalone module which accepts float input
        and produces float output.
    """
    return _convert_fx(graph_module, debug, convert_custom_config_dict, is_standalone_module=True)
