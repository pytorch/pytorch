# FX Graph Mode Quantization Design Doc
High Level FX Graph Mode Quantization Flow
```
float_model            QConfigMapping           BackendConfig
    \                          |                        /
     \                         |                      /
      \                        |                    /
(prepare_fx/prepare_qat_fx)                        /
—-------------------------------------------------------
|                         Fuse                         |
|                  QAT Module Swap                     |
|                 Insert Observers                     |
—-------------------------------------------------------
                              |
                      Calibrate/Train
                              |
(convert_fx)                  |
—--------------------------------------------------------
|                         Convert                       |
|                        Lowering                       |
—--------------------------------------------------------
                              |
                       Quantized Model

```
Please refer to [TODO: link] for definitions of terminologies.

## Overview
The FX graph representation is pretty close to python/eager mode, it preserves many python/eager mode constructs like modules, functionals, torch ops, so overall the implementation reuses some of building blocks and utilities from eager mode quantization, this includes the QConfig, QConfig propagation (might be removed), fused modules, QAT module, quantized modules, QAT module swapping utility. Also the overall flow exactly matches eager mode quantization, the only difference is that the transformations like fusion, inserting stubs are fully automated and controlled by QConfigMapping and BackendConfig.

## High Level Flow with Simple Example

`prepare_fx`:
```
Floating Point Model --> (1.1 `_fuse_fx`) --> Fused Model
                     --> (1.2 QAT Module Swap) --> Model with QAT modules
                     --> (1.3 Insert Observers) --> Prepared Model
```

`convert_fx`:
```
Prepared Model --> (2.1 `convert_to_reference`) --> Reference Quantized Model
               --> (2.2 Lower to Native Backend) --> Quantized Model
```

In the following, I’ll first have a detailed description for each step, and then talk about the corresponding settings in BackendConfig. We’ll follow the terminologies defined in (draft) README.md of quantization syntax transforms in this doc.

### 0. Original Model

```
class LinearReLUModule(torch.nn.Module):
   def __init__(self):
       super().__init__()
       self.linear = torch.nn.Linear(5, 10).float()
       self.relu = torch.nn.ReLU()

   def forward(self, x):
       return self.relu(self.linear(x))
```

### 1.1 Fusion
```
fused: GraphModule(
  (linear): LinearReLU(
    (0): Linear(in_features=5, out_features=10, bias=True)
    (1): ReLU()
  )
)

def forward(self, x):
    linear = self.linear(x);  x = None
    return linear
```

What we did in this example are:

* Identify (Linear - ReLU) subgraph by searching through the model graph
* For each of the identified subgraph, we replace the `root_node` (typically the weighted module in the pattern, like Linear), with a fused module by calling the fuser_method for this pattern, a fused module is a sequential of a few modules, e.g. nni.LinearReLU is a sequential of linear and relu module

`backend_config` configurations relevant to this step are:

```
def fuse_linear_relu(is_qat, linear, relu):
    return nni.LinearReLU(linear, relu)

BackendPatternConfig((torch.nn.Linear, torch.nn.ReLU))
    .set_fuser_method(fuse_linear_relu)
    ._set_root_node_getter(my_root_node_getter)
    ._set_extra_inputs_getter(my_extra_inputs_getter)
```


`BackendPatternConfig` takes in a pattern that specifies the fusion pattern that we want to search for, pattern format can be found in https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/backend_config/README.md

`set_dtype_configs`: dtype_configs are used to check against the qconfig for the pattern, to see if the qconfig is supported in the target backend or not. Currently it’s not used in fusion, but we can add this check in the future, or remove this and always fuse these patterns.
`set_fuser_method`: specifies the fuser method to use for the pattern, a fuser method will take the matched object and fuse them into a fused module.
`_set_root_node_getter`: sets a function that takes a node pattern and returns the root node in the pattern.
`_set_extra_inputs_getter`: all input args of root node will be copied over to fused module, if there are extra inputs, this function will return a list of extra inputs given the pattern.

Example usage of `root_node_getter` and `extra_input_getter`: https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6

### 1.2 QAT Module Swap
```
GraphModule(
  (linear): LinearReLU(
    in_features=5, out_features=10, bias=True
    (weight_fake_quant): MinMaxObserver(min_val=inf, max_val=-inf)
  )
)

def forward(self, x):
    linear = self.linear(x);  x = None
    return linear
```

In this step we swap the fused module to qat module, for example, swap nn.intrinsic.LinearReLU instances to nn.intrinsic.qat.LinearReLU module where we fake quantize the weight of linear.
For modules that has corresponding QAT modules we’ll call eager mode `convert` function with a mapping from float module to QAT module which will swap all float module (and fused module) with QAT module, this step is exactly the same as eager mode quantization, just called inside the `prepare_fx/prepare_qat_fx` function.

`backend_config` configurations relevant in this step are:
```
BackendPatternConfig(nni.LinearReLU)
    .set_qat_module(nniqat.LinearReLU)
```

The pattern used to initialize BackendPatternConfig is the class type for original or fused floating point module class.
`set_qat_module` sets the qat module class corresponding to the module class specified in the pattern.

### 1.3 QuantDeQuantStub and Observer/FakeQuantize Insertion
```
GraphModule(
  (activation_post_process_0): MinMaxObserver(min_val=inf, max_val=-inf)
  (linear): LinearReLU(
    (0): Linear(in_features=5, out_features=10, bias=True)
    (1): ReLU()
  )
  (activation_post_process_1): MinMaxObserver(min_val=inf, max_val=-inf)
)

def forward(self, x):
    activation_post_process_0 = self.activation_post_process_0(x);  x = None
    linear = self.linear(activation_post_process_0);  activation_post_process_0 = None
    activation_post_process_1 = self.activation_post_process_1(linear);  linear = None
    return activation_post_process_1
```

Note: activation_post_process_0 and activation_post_process_1 will be updated with QuantDeQuantStub

QuantDeQuantStubs are inserted based on the `qconfig_mapping` provided by users. Also we have a backend_config that specifies the configs that are supported by the backend. In this step, we will
* Check if `qconfig_mapping` is compatible with `backend_config` or not, if user requested a qconfig that is not compatible with `backend_config`, we’ll not insert observers for the operator, the config would just be ignored.
* Insert observer for the input and output of the subgraph, based on the `qconfig_mapping` (what user requested) and the `backend_config` (how the operator should be observed in a backend).

Detailed walkthrough for this step in `prepare_qat_fx` (inserting QDQStub and FakeQuantize modules):
Note: We could also insert QStub and DQStub in this step when users request to change the interface dtype for the model, standalone module or custom modules.
```
# fused and qat swapped model
# graph 1:
input - qat_linear_relu - output
              |
          FakeQuantize
(need to be updated with QDQStub + FakeQuantize)
              |
           weight

# qconfig_mapping (simplified, shown as dict)
{'qat_linear_relu': QConfig(
  weight=MinMaxObserver.with_args(dtype=torch.qint8),
  activation=HistogramObserver.with_args(dtype=torch.quint8),
)}

# backend_config (simplified)
{
  'pattern': nnqat.LinearReLU,
  'dtype_configs': [{input: torch.quint8, output: torch.quint8, weight: torch.qint8}],
}
```

step 1: assign qconfig to each op (please see [TODO: link] for details)

step 2: determine which qconfigs are valid according to the backend configuration (please see [TODO: link] for details)
(we should add a warning here)

step 3: for subgraphs with validated qconfigs, insert qstub/dqstub/qdqstub needed

To talk about what happens in this step, let’s first define some terms. Let’s view the computation graph we showed above as a Graph consists of nodes and edges, each node here will be an FX Node that represents some computation, for example linear, and each edge will be a connection between two nodes, and each edge can both be viewed as the output of the previous Node or the input of the next Node.

The end goal for this step is to insert QDQStubs at edges so that we produce a graph of quantized reference model when each QDQStub represents a quantize operator followed by a dequantize operator.

```
# graph 2:
input - QDQStub1 (FakeQuantize) - qat_linear_relu - QDQStub2 (FakeQuantize) - output
                                      |
                                FakeQuantize
                  (need to be updated with QDQStub + FakeQuantize)
                                      |
                                    weight
```
Note: weight + FakeQuantize is a part of qat_linear_relu

The overall logic to insert QDQStub1 and QDQStub2 inplace is the following:
0. For each node in the original graph, we compute the target_dtype for input and output for it based on qconfig, for graph1, configured with qconfig_mapping, we have:
```
# node_name_to_target_dtype_info =
# {
#     # this is placeholder node in FX Graph
#     "input" : {"input_activation": torch.float32, "output_activation": torch.float32},
#     "qat_linear_relu": {"input_activation": torch.quint8, "output_activation": torch.quint8, "weight": ...}
#     # this is the return node in FX Graph
#     "output": {"input_activation": torch.float32, "output_activation": torch.float32}
# }
```
Note: this map is generated before we insert qdqstub to graph1, and will not change in the process.

1. Inserting QDQStub1 (for input of qat_linear_relu)
   We need to look at the edge between `input` Node and `qat_linear_relu` Node here, we need to decide if we need to insert a
   QDQStub at this edge, which could serve as an input argument for `qat_linear_relu` Node (and also output for `input` Node)
   The way we decide if we want to insert QDQStub here is to figure out

   (1). The target dtype for output of `input` Node, which is torch.float32

   (2). The target dtype for input of `qat_linear_relu` Node, which is torch.quint8
   There is a mismatch here and (2) is a quantized dtype, so we need to insert QDQStub at the edge.

   We also need to attach observer/fakequant module to the QDQStub we inserted here.
2. Insert QDQStub2 (for output of qat_linear_relu)
   The logic for inserting QDQStub for output is much easier, since we assume all modules/functions in the graph produce fp32 output
   by default (we can have additional checks and extend this to work for other dtypes after we have type inference ready),
   we just need to look at the target output dtype for qat_linear_relu Node, and if it is a quantized dtype (quint8, qint8, float16),
   we would insert a QDQStub here.

Questions: How to avoid inserting duplicate QDQStubs?
e.g. when we have a single input being used by multiple ops:
```
input — linear1 —-
     \--- linear2 —
```
how do we make sure we only insert one QDQStub for input of both linear1 and linear2?
```
input - QDQStub — linear1 -
             \ —- linear2 -
```

The way we do it right now is before we insert QDQStub, we look at all users of `input` Node here and make sure there is no QDQStubs
with the same target_dtype, that is, if we already inserted a QDQStub with dtype quint8 for linear1, and linear2 is also connected to it, if we request another QDQStub with dtype quint8 when processing linear2 Node, we’ll detect that the desired QDQStub already exists and do nothing

Question: What is the logic for keeping output to be float32?
Let’s say the output of `qat_linear_relu` Node is configured as float32, both in qconfig_mapping and backend_config:
```
# qconfig_mapping (simplified, shown as dict)
{'qat_linear_relu': QConfig(
  weight=MinMaxObserver.with_args(dtype=torch.qint8),
  input_activation=HistogramObserver.with_args(dtype=torch.quint8),
  output_activation=PlaceholderObserver.with_args(dtype=torch.float32),
)}

# backend_config (simplified)
{
  'pattern': nnqat.LinearReLU,
  'dtype_configs': [{input: torch.quint8, output: torch.float32, weight: torch.qint8}],
}
```

What we’ll do here is when we are trying to insert output QDQStub for `qat_linear_relu`, we look at the target output dtype for this node (node_name_to_target_dtype_info["qat_linear_relu"]["output_activation"], and find that it is float, which is not a quantized dtype, so
will do nothing here.
Note that this does not prevent other operators following `qat_linear_relu` to insert a QDQStub at the output of `qat_linear_relu`, since we are dealing with an `edge` of the graph here, and an `edge` is connected to two nodes, which means
the output of `qat_linear_relu` will also be the input of a node following `qat_linear_relu`.

`backend_config` configurations used in this step:
```
BackendConfig(nniqat.LinearReLU)
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)
    .set_dtype_configs([
        DTypeConfig(input_dtype=torch.quint8, output_dtype = torch.quint8, weight_dtype = torch.qint8, bias_dtype = torch.float32)]
    )
```

Pattern in this case is the same as before, it defines the pattern for the subgraph we are dealing with

`set_observation_type`: sets the observation type for the patter, currently only two types:

`OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT` means the output observer instance will be different from the input, which is the most common type of observer placement.

`OUTPUT_SHARE_OBSERVER_WITH_INPUT` means the output observer is shared with input, they will be the same instance. This is useful for operators like cat.

`set_dtype_configs`: sets a list of supported (activation, weight, bias, etc.) dtype combinations for qconfigs for the pattern. Note that we represent different modes of quantization (static/dynamic/`weight_only`) purely through this combination, for example, fbgemm static quantization can be represented as:
```
{
  "input_activation": torch.quint8,
  "weight": torch.qint8,
  "output_activation": torch.quint8
}
```

Note: the dtype config will be used to configure the support for dynamic quantization as well

Note: we may extend this to support more fine grained configurations of args, kwargs, attributes and outputs in the future

Note: we are referring to observer here, which is an implementation detail, we can change this to talk about quantization parameters instead, e.g. `QParamsType.OUTPUT_USE_DIFFERENT_QPARAMS_AS_INPUT` and `QParamsType.OUTPUT_USE_SAME_QPARAMS_AS_INPUT`

### 2. Calibration/Training
After we insert observers, we run the model to calibrate observers or to fine tune. This step is identical to eager mode quantization. After that the observer/fakequantize modules contain sufficient information to determine quantization parameters according to the observed data.

### 3.1 Conversion to Reference Quantized Model
```
quantized: GraphModule(
  (linear): LinearReLU(
    (0): QuantizedLinear(Reference)(in_features=5, out_features=10, bias=True)
    (1): ReLU()
  )
)

def forward(self, x):
    linear_input_scale_0 = self.linear_input_scale_0
    linear_input_zero_point_0 = self.linear_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
    dequantize = quantize_per_tensor.dequantize();  quantize_per_tensor = None
    linear = self.linear(dequantize);  dequantize = None
    linear_scale_0 = self.linear_scale_0
    linear_zero_point_0 = self.linear_zero_point_0
    quantize_per_tensor_1 = torch.quantize_per_tensor(linear, linear_scale_0, linear_zero_point_0, torch.quint8);  linear = linear_scale_0 = linear_zero_point_0 = None
    dequantize_1 = quantize_per_tensor_1.dequantize();  quantize_per_tensor_1 = None
    return dequantize_1
```

After we insert observers, we’ll need to convert the model to a reference quantized model. Reference quantized model is a model that uses reference patterns to represent quantized operators, this serves as the standard interface for quantized operators between PyTorch quantization and backend lowering passes. For more details, please take a look at this [RFC](https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md). This pass is pretty straightforward, what we do is:

(1). for each QDQStub (attached with Observer for FakeQuantize modules) in the graph, we'll convert it to calls to quantize and dequantize functions based on the attributes of attached Observer and FakeQuantize modules (e.g. qscheme, dtype etc.)

(2). for weighted modules like linear/conv, we convert them to corresponding reference quantized module.

Example:
```
# graph 1
input - QDQStub1 (FakeQuantize) - qat_linear_relu - QDQStub2 (FakeQuantize) - output
                                      |
                                FakeQuantize
                  (need to be updated with QDQStub + FakeQuantize)
                                      |
                                    Weight

Note: weight + FakeQuantize is a part of qat_linear_relu module

# graph 2
input - quantize - dequantize - reference_linear_relu - quantize - dequantize - output
                                        |
                                   dequantize
                                        |
                                    quantize
                                        |
                                      weight
```
Note: weight + quantize + dequantize is a part of reference_linear_relu module

To decide which quantize node we want to use, we’ll look at:

(1). dtype of attached Observer/FakeQuantize module

(2). qscheme of attached Observer/FakeQuantize module

(3). (optionally) other attributes of attached Observer/FakeQuantize module

The quantize operator we can choose from right now are: (quantize_per_tensor, quantize_per_channel, to, quantize_per_tensor_dynamic)

```
backend_config configurations used in this step:
BackendConfig(nniqat.LinearReLU)
    .set_root_module(nn.Linear)
    .set_reference_quantized_module_for_root(nnqr.Linear)
    .set_fused_module(nni.LinearReLU)
```

Pattern in this case is the same as before, it defines the pattern for the subgraph we are dealing with

`set_root_module`: Sets a module class for the root of the pattern, e.g. nn.Linear for a nni.LinearReLU/nniqat.LinearReLU, used to identify the modules that needs to be swapped to reference quantized module

`set_reference_quantized_module_for_root`: Sets the corresponding reference quantized module class for root module class, e.g. when root_module is nn.Linear, this will be nn.quantized.reference.Linear, used to swap the root module to be a reference quantized module.

Note: we are only swapping `root_module` here, for example, in the current example, the original module is `nniqat.LinearReLU`, when we are converting weight modules(step (2)), we first convert `nniqat.LinearReLU` to a float module, in this case, the fused LinearReLU module: `nni.LinearReLU`, and then swap the root_module (`nn.Linear`) with reference quantized module (`nnqr.Linear`), so we end up with a `nni.LinearReLU` module, which is a sequential module of a `nnqr.Linear` and `nn.ReLU`.

Basically, the corresponding reference quantized module for both `nniqat.LinearReLU` and `nni.LinearReLU` would be a `nni.LinearReLU` Sequential module (originally `nn.Linear` + `nn.ReLU`) with `nn.Linear` being replaced by `nnqr.Linear`: `nni.LinearReLU(nnqr.Linear, nn.ReLU)`.

`set_fused_module`: This is the corresponding fused module class for the pattern, used to identify fused modules that needs to be converted to reference quantized module

### 3.2 Lower to PyTorch Native Backend
```
GraphModule(
  (linear): QuantizedLinearReLU(in_features=5, out_features=10, scale=1.0, zero_point=0, qscheme=torch.per_tensor_affine)
)

def forward(self, x):
    linear_input_scale_0 = self.linear_input_scale_0
    linear_input_zero_point_0 = self.linear_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(x, linear_input_scale_0, linear_input_zero_point_0, torch.quint8);  x = linear_input_scale_0 = linear_input_zero_point_0 = None
    linear = self.linear(quantize_per_tensor);  quantize_per_tensor = None
    dequantize_1 = linear.dequantize();  linear = None
    return dequantize_1
```

Currently, PyTorch has native quantized backends: fbgemm and qnnpack, so we need a lowering pass to lower the reference quantized model to a model that is using native quantized operators in PyTorch. What this pass did is

1. Recognize the reference patterns like: "dequantize - `float_op` - quantize" in the graph and replace them with the quantized modules (under torch.nn.quantized namespace) or operators (under torch.ops.quantized namespace, or torch namespace)
In general there are three types of patterns:

* Static quantization:
```
dequantize -> float_op -> quantize_per_tensor
```

* Dynamic quantization:
```
quantize_per_tensor_dynamic -> dequantize -> float_op
```

* Weight only quantization:
```
                                       input - float_op - output
      weight - quantize_per_tensor - dequantize /
```

2. Prepack and fold the weights for quantized linear and quantized conv operator
3. The lowering pass is also going to keep some patterns for quantized operators unfused, since user may explicitly request some operators to stay in float by configuring the qconfig to be None

There are no configurations related to lowering in `backend_config` since it is backend developer’s responsibility to implement lowering pass and each of the backend developers may have their own configurations. So from end to end, `backend_config` and together with qconfig_mapping controls what Reference Quantized Model is produced by FX Graph Mode Quantization, not lowered model.

However, for some operator based backends, like the current pytorch native backends including fbgemm and qnnpack. We could interpret `backend_config` in terms of configurations for operators as well. e.g. configuring `input_dtype=quint8`, `weight_dtype=qint8`, `output_dtype=torch.quint8` for nn.Linear is saying that the quantized linear will take a `quint8` activation and `qint8` weight as input and outputs a `quint8` activation. But there is no guarantee that this interpretation will always work in the future, especially when we add new flavors of quantized operators.

## Extensibility

FX graph mode quantization can be extended to work with different backends, which may have different sets of supported quantized operator patterns and different requirements for each pattern. For more detail, please refer to the [BackendConfig README](/torch/ao/quantization/backend_config/README.md).
