## BackendConfig Overview

BackendConfig allows PyTorch quantization to work with different backend or kernel libraries. These backends may have different sets of supported quantized operator patterns, and the same operator patterns may require different handling across different backends. To make quantization work with different backends and allow maximum flexibility, we strived to make all the parts of the quantization flow configurable with BackendConfig. Currently, it is only used by FX graph mode quantization. For more details on how it integrates with the FX graph mode quantization flow, refer to this [README](/torch/ao/quantization/fx/README.md).

BackendConfig configures quantization behavior in terms of operator patterns. For each operator pattern, we need to specify what the supported data types are for the input and output activations, weights, and biases, and also specify the QAT modules, the reference quantized modules etc., which will be used in module swapping during the quantization passes.

Quantized backends can have different support in terms of the following aspects:
* Quantization scheme (symmetric vs asymmetric, per-channel vs per-tensor)
* Data type (float32, float16, int8, uint8, bfloat16, etc.) for input/output/weight/bias
* Quantized (and fused) mapping: Some quantized operators may have different numerics compared to a naive (dequant - float_op - quant) reference implementation. For weighted operators, such as conv and linear, we need to be able to specify custom reference modules and a mapping from the float modules
* QAT mapping: For weighted operators, we need to swap them with the Quantization Aware Training (QAT) versions that add fake quantization to the weights

As an example, here is what fbgemm looks like:
|                                           | fbgemm                                                                |
|-------------------------------------------|-----------------------------------------------------------------------|
| Quantization Scheme                       | activation: per tensor, weight: per tensor or per channel             |
| Data Type                                 | activation: quint8 (with qmin/qmax range restrictions), weight: qint8 |
| Quantized and Fused Operators and Mapping | e.g. torch.nn.Conv2d -> torch.ao.nn.quantized.reference.Conv2d        |
| QAT Module Mapping                        | e.g. torch.nn.Conv2d -> torch.ao.nn.qat.Conv2d                        |

Instead of hardcoding the fusion mappings, float to reference quantized module mappings, fusion patterns etc., we will derive everything from the BackendConfig throughout the code base. This allows PyTorch Quantization to work with all first-party (fbgemm and qnnpack) and third-party backends (TensorRT, executorch etc.) that may differ from native backends in different aspects. With the recent addition of xnnpack, integrated as part of the qnnpack backend in PyTorch, the BackendConfig is needed to define the new constraints required for xnnpack quantized operators.

## Pattern Specification

The operator patterns used in BackendConfig are float modules, functional operators and pytorch operators specified in reverse order:
```
operator = module_type | functional | torch op | native op | MatchAllNode
Pattern = (operator, Pattern, Pattern, ...) | operator
```
where the first item for each Pattern is the operator, and the rest are the patterns for the arguments of the operator.
For example, the pattern (nn.ReLU, (operator.add, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d))) would match the following graph:
```
tensor_1            tensor_2
 |                    |
 *(MatchAllNode)  nn.Conv2d
 |                    |
 |             nn.BatchNorm2d
 \                  /
  -- operator.add --
         |
      nn.ReLU
```

During prepare and convert, we’ll match the last node, which will be the anchor point of the match, and we can retrieve the whole graph by tracing back from the node. E.g. in the example above, we matched the `nn.ReLU` node, and `node.args[0]` is the `operator.add` node.

## BackendConfig Implementation

The BackendConfig is comprised of a list of BackendPatternConfigs, each of which define the specifications and the requirements for an operator pattern. Here is an example usage:

```
import torch
from torch.ao.quantization.backend_config import BackendConfig, BackendPatternConfig, DTypeConfig, ObservationType
from torch.ao.quantization.fuser_method_mappings import reverse_sequential_wrapper2

weighted_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_type=torch.float)

linear_config = BackendPatternConfig(torch.nn.Linear) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Linear) \
    .set_qat_module(torch.ao.nn.qat.Linear) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

conv_relu_config = BackendPatternConfig((torch.nn.ReLU, torch.nn.Conv2d)) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_fuser_method(reverse_sequential_wrapper2(torch.ao.nn.intrinsic.ConvReLU2d))

backend_config = BackendConfig("my_backend") \
    .set_backend_pattern_config(linear_config) \
    .set_backend_pattern_config(conv_relu_config)
```

### Observer Insertion

Relevant APIs:
* `set_observation_type`

During the prepare phase, we insert observers (or QuantDeQuantStubs in the future) into the graph for this operator pattern based on the observation type, which specifies whether to use different observers for the inputs and the outputs of the pattern. For more detail, see `torch.ao.quantization.backend_config.ObservationType`.

### Reference Quantized Patterns

Relevant APIs:
* `set_root_module`
* `set_reference_quantized_module`

During the convert phase, when we construct the reference quantized model, the root modules (e.g. `torch.nn.Linear` for `nni.LinearReLU` or `nniqat.LinearReLU`) will be swapped to the corresponding reference quantized modules (e.g. `torch.ao.nn.reference.Linear`). This allows custom backends to specify custom reference quantized module implementations to match the numerics of their lowered operators. Since this is a one-to-one mapping, both the root module and the reference quantized module must be specified in the same BackendPatternConfig in order for the conversion to take place.

### Fusion

Relevant APIs:
* `set_fuser_method`
* `set_fused_module`
* `_set_root_node_getter`
* `_set_extra_inputs_getter`

As an optimization, operator patterns such as (`torch.nn.ReLU`, `torch.nn.Linear`) may be fused into `nni.LinearReLU`. This is performed during the prepare phase according to the function specified in `set_fuser_method`, which replaces the pattern with the fused module. During the convert phase, these fused modules (identified by `set_fused_module`) will then be converted to the reference quantized versions of the modules.

In FX graph mode quantization, we replace the corresponding nodes in the graph using two helper functions set by the user: `root_node_getter`, which returns the root node (typically the weighted module in the pattern like `torch.nn.Linear`) to replace the matched pattern in the graph, and `extra_inputs_getter`, which returns a list of extra input arguments that will be appended to the existing arguments of the fused module (copied over from the root node). See [this snippet](https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6) for an example usage.

### Data Type Restrictions

Relevant APIs:
* `add_dtype_config`
* `set_dtype_configs`

DTypeConfig specifies a set of supported data types for input/output/weight/bias along with the associated constraints, if any. There are two ways of specifying `input_dtype`, `output_dtype`, and `weight_dtype`, as simple `torch.dtype`s or as `DTypeWithConstraints`, e.g.:

```
import torch
from torch.ao.quantization.backend import DTypeConfig, DTypeWithConstraints

dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float)

dtype_config_with_constraints = DTypeConfig(
    input_dtype=DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=255,
        scale_min_lower_bound=2 ** -12,
    ),
    output_dtype=DTypeWithConstraints(
        dtype=torch.quint8,
        quant_min_lower_bound=0,
        quant_max_upper_bound=255,
        scale_min_lower_bound=2 ** -12,
    ),
    weight_dtype=DTypeWithConstraints(
        dtype=torch.qint8,
        quant_min_lower_bound=-128,
        quant_max_upper_bound=127,
        scale_min_lower_bound=2 ** -12,
    ),
    bias_dtype=torch.float)
```

During the prepare phase of quantization, we will compare the data types specified in these DTypeConfigs to the ones specified in the matching QConfig for a given operator pattern. If the data types do not match (or the constraints are not satisfied) for all the DTypeConfigs specified for the operator pattern, then we will simply ignore the QConfig and skip quantizing this pattern.

#### Quantization range

The user's QConfig may specify `quant_min` and `quant_max`, which are min and max restrictions on the quantization values. Here we set the lower bound for the `quant_min` and then upper bound for the `quant_max` to represent the limits of the backend. If a QConfig exceeds these limits in either direction, it will be treated as violating this constraint.

#### Scale range

Similarly, the user's QConfig may specify a minimum value for the quantization scale (currently exposed as `eps` but will change in the future to better reflect the semantics). Here we set the lower bound for the `scale_min` to represent the limits of the backend. If a QConfig's min scale value falls below this limit, the QConfig will be treated as violating this constraint. Note that `scale_max_upper_bound` is currently not used, because there is no corresponding mechanism to enforce this on the observer yet.
