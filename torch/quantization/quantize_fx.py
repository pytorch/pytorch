import torch
from torch.fx import GraphModule  # type: ignore
from torch.fx.symbolic_trace import Tracer  # type: ignore
from torch.fx.node import Target, Node, Argument  # type: ignore
from .fx import Fuser  # noqa: F401
from .fx import Quantizer  # noqa: F401
from .fx.utils import graph_pretty_str  # noqa: F401
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from .fx.graph_module import ObservedGraphModule, QuantizedGraphModule
from torch.nn.intrinsic import _FusedModule
from typing import Dict, Any, List, Callable, Tuple, Optional

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

class Scope(object):
    """ Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example:
    class Sub(torch.nn.Module):
        def forward(self, x):
            # This will be a call_method Node in GraphModule,
            # scope for this would be (module_path="sub", module_type=Sub)
            return x.transpose(1, 2)

    class M(torch.nn.Module):
        def __init__(self):
            self.sub = Sub()

        def forward(self, x):
            # This will be a call_method Node as well,
            # scope for this would be (module_path="", None)
            x = x.transpose(1, 2)
            x = self.sub(x)
            return x

    """
    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type

class ScopeContextManager(object):
    """ A context manager to track the Scope of Node during symbolic
    tracing.
    When entering a forward function of a Module, we'll update the scope information of
    the current module, and when we exit, we'll restore the previous scope information.
    """
    def __init__(
            self,
            scope: Scope,
            current_module: torch.nn.Module,
            current_module_path: str):
        super().__init__()
        self.prev_module_type = scope.module_type
        self.prev_module_path = scope.module_path
        self.scope = scope
        self.scope.module_path = current_module_path
        self.scope.module_type = type(current_module)

    def __enter__(self):
        return

    def __exit__(self, *args):
        self.scope.module_path = self.prev_module_path
        self.scope.module_type = self.prev_module_type
        return


class QuantizationTracer(Tracer):
    def __init__(
            self,
            skipped_module_names: List[str],
            skipped_module_classes: List[Callable]):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type of top level
        # module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.node_name_to_scope : Dict[str, Tuple[str, type]] = {}

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        return (m.__module__.startswith("torch.nn") and
                not isinstance(m, torch.nn.Sequential)) or \
            module_qualified_name in self.skipped_module_names or \
            type(m) in self.skipped_module_classes or \
            isinstance(m, _FusedModule)

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        module_qualified_name = self.path_of_module(m)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, m, module_qualified_name):
            return super().call_module(m, forward, args, kwargs)

    def create_node(self, kind : str, target : Target,
                    args : Tuple[Argument, ...], kwargs : Dict[str, Argument], name : Optional[str] = None,
                    type_expr : Optional[Any] = None) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (self.scope.module_path, self.scope.module_type)
        return node

def _prepare_fx(model: torch.nn.Module, qconfig_dict: Any,
                prepare_custom_config_dict: Dict[str, Any] = None,
                is_standalone_module: bool = False) -> ObservedGraphModule:
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
    tracer = QuantizationTracer(
        skipped_module_names, skipped_module_classes)
    graph_module = GraphModule(model, tracer.trace(model))
    graph_module = _fuse_fx(graph_module, prepare_custom_config_dict)
    quantizer = Quantizer()
    prepared = quantizer.prepare(
        graph_module,
        qconfig_dict,
        tracer.node_name_to_scope,
        prepare_custom_config_dict=prepare_custom_config_dict,
        is_standalone_module=is_standalone_module)

    preserved_attributes = prepare_custom_config_dict.get("preserved_attributes", [])
    for attr_name in preserved_attributes:
        setattr(prepared, attr_name, getattr(model, attr_name))
    return prepared

def _prepare_standalone_module_fx(
        model: torch.nn.Module,
        qconfig_dict: Any,
        prepare_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" [Internal use only] Prepare a standalone module, so that it can be used when quantizing the
    parent module.
    standalone_module means it a submodule that is not inlined in parent module,
        and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module

    Returns:
        model(GraphModule): prepared standalone module
        attributes:
            _standalone_module_input_quantized_idxs(List[Int]): a list of
                indexes for the graph input that is expected to be quantized,
                same as input_quantized_idxs configuration provided
                for the standalone module
            _standalone_module_output_quantized_idxs(List[Int]): a list of
                indexs for the graph output that is quantized
                same as input_quantized_idxs configuration provided
                for the standalone module
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
        prepare_custom_config_dict: Dict[str, Any] = None) -> ObservedGraphModule:
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

        # Attributes that are not used in forward function will
        # be removed when constructing GraphModule, this is a list of attributes
        # to preserve as an attribute of the GraphModule even when they are
        # not used in the code
        "preserved_attributes": ["preserved_attr"],
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
        prepare_custom_config_dict: Dict[str, Any] = None) -> ObservedGraphModule:
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
        graph_module: GraphModule, is_reference: bool,
        convert_custom_config_dict: Dict[str, Any] = None,
        is_standalone_module: bool = False,
        _remove_qconfig: bool = True) -> QuantizedGraphModule:
    """ `is_standalone_module`: see docs in :func:`~torch.quantization.prepare_standalone_module_fx`
    """
    if convert_custom_config_dict is None:
        convert_custom_config_dict = {}

    _check_is_graph_module(graph_module)

    quantizer = Quantizer()
    quantized = quantizer.convert(graph_module, is_reference, convert_custom_config_dict,
                                  is_standalone_module, _remove_qconfig=_remove_qconfig)

    preserved_attributes = convert_custom_config_dict.get("preserved_attributes", [])
    for attr_name in preserved_attributes:
        setattr(quantized, attr_name, getattr(graph_module, attr_name))
    return quantized

def convert_fx(
        graph_module: GraphModule, is_reference: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None,
        _remove_qconfig: bool = True) -> QuantizedGraphModule:
    r""" Convert a calibrated or trained model to a quantized model
    Args:
        `graph_module`: A prepared and calibrated/trained model (GraphModule)
        `is_reference`: flag for whether to produce a reference quantized model,
        which will be a common interface between pytorch quantization with
        other backends like accelerators
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
          },

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
          },

          # Attributes that are not used in forward function will
          # be removed when constructing GraphModule, this is a list of attributes
          # to preserve as an attribute of the GraphModule even when they are
          # not used in the code
          "preserved_attributes": ["preserved_attr"],
        }
        `_remove_qconfig`: Option to remove the qconfig attributes in the model after convert.

    Return:
        A quantized model (GraphModule)

    Example:
    ```python
    # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
    quantized_model = convert_fx(prepared_model)
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_fx")
    return _convert_fx(graph_module, is_reference, convert_custom_config_dict, _remove_qconfig=_remove_qconfig)

def _convert_standalone_module_fx(
        graph_module: GraphModule, is_reference: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None) -> QuantizedGraphModule:
    r""" [Internal use only] Convert a model produced by :func:`~torch.quantization.prepare_standalone_module_fx`
    and convert it to a quantized model

    Returns a quantized standalone module, whether input/output is quantized is
    specified by prepare_custom_config_dict, with
    input_quantized_idxs, output_quantized_idxs, please
    see docs for prepare_fx for details
    """
    return _convert_fx(graph_module, is_reference, convert_custom_config_dict, is_standalone_module=True)
