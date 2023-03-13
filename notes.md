# PyTorch 2.0 Release notes

# Highlights

- &lt;Summary to be provided>

# Backwards Incompatible changes

### **Drop support for Python versions <= 3.7 (#93155)**

Previously the minimum supported version of Python for PyTorch was 3.7. This PR updates the minimum version to require 3.8 in order to install PyTorch. See [Hardware / Software Support ](https://github.com/pytorch/pytorch/blob/893aa5df3f2a475c91ea8eadb1353812e52fb227/RELEASE.md#python) for more information.

### **Drop support for CUDA 10 (#89582)**

This PR updates the minimum CUDA version to 11.0. See the [getting-started](https://pytorch.org/get-started/locally/) for installation or [building from source](https://github.com/pytorch/pytorch#from-source) for more information.

### **Gradients are now set to `None` instead of zeros by default in `torch.optim.*.zero_grad()` and `torch.nn.Module.zero_grad()` (#92731)**

This changes the default behavior of `zero_grad()` to zero out the grads by setting them to `None` instead of zero tensors. In other words, the `set_to_none` kwarg is now `True` by default instead of `False`. Setting grads to `None` reduces peak memory usage and increases performance. This will break code that directly accesses data or does computation on the grads after calling `zero_grad()` as they will now be `None`. To revert to the old behavior, pass in `zero_grad(set_to_none=False)`.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
>>> import torch
>>> from torch import nn
>>> module = nn.Linear(2,22)
>>> i = torch.randn(2, 2, requires_grad=True)
>>> module(i).sum().backward()
>>> module.zero_grad()
>>> module.weight.grad == None
False
>>> module.weight.grad.data
tensor([[0., 0.],
        [0., 0.]])
>>> module.weight.grad + 1.0
tensor([[1., 1.],
        [1., 1.]])
```

</td>
<td>

```Python
>>> import torch
>>> from torch import nn
>>> module = nn.Linear(5, 5)
>>> i = torch.randn(2, 5, requires_grad=True)
>>> module(i).sum().backward()
>>> module.zero_grad()
>>> module.weight.grad == None
True
>>> module.weight.grad.data
AttributeError: 'NoneType' object has no attribute 'data'
>>> module.weight.grad + 1.0
TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'
```

</td>
</tr>
</table>

### **Update `torch.tensor` and `nn.Parameter` to serialize all their attributes (#88913)**

Any attribute stored on `torch.tensor` and `torch.nn.Parameter` will now be serialized. This aligns the serialization behavior of `torch.nn.Parameter`, `torch.Tensor` and other tensor subclasses

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
# torch.Tensor behavior
>>> a = torch.Tensor()
>>> a.foo = 'hey'

>>> buffer = io.BytesIO()
>>> torch.save(a, buffer)
>>> buffer.seek(0)
>>> b = torch.load(buffer)

>>> print(a.foo)
hey
>>> print(b.foo)
AttributeError: 'Tensor' object has no attribute 'foo'

# torch.nn.Parameter behavior
>>> a = nn.Parameter()
>>> a.foo = 'hey'

>>> buffer = io.BytesIO()
>>> torch.save(a, buffer)
>>> buffer.seek(0)
>>> b = torch.load(buffer)

>>> print(a.foo)
hey
>>> print(b.foo)
AttributeError: 'Parameter' object has no attribute 'foo'
```

</td>
<td>

```Python
# torch.Tensor behavior
a = torch.Tensor()
a.foo = 'hey'

>>> buffer = io.BytesIO()
>>> torch.save(a, buffer)
>>> buffer.seek(0)
>>> b = torch.load(buffer)
>>> print(a.foo)
hey
>>> print(b.foo)
hey

# torch.nn.Parameter behavior
>>> a = nn.Parameter()
>>> a.foo = 'hey'

>>> buffer = io.BytesIO()
>>> torch.save(a, buffer)
>>> buffer.seek(0)
>>> b = torch.load(buffer)
>>> print(a.foo)
hey
>>> print(b.foo)
hey
```

</td>
</tr>
</table>

If you have an attribute that you don't want to be serialized you should not store it as an attribute on tensor or Parameter but instead it is recommended to use `torch.utils.weak.WeakTensorKeyDictionary`

```Python
>>> foo_dict = weak.WeakTensorKeyDictionary()
>>> foo_dict[a] = 'hey'
>>> print(foo_dict[a])
hey
```

### **Algorithms `{Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, NAdam, RAdam, RMSProp, RProp, SGD}` default to faster `foreach` implementation when on CUDA + differentiable=`False`**

When applicable, this changes the default behavior of `step()` and anything that calls into `adadelta(...)`, `adagrad(...)`, `adam(...)`, `adamax(...)`, `adamw(...)`, `asgd(...)`, `nadam(...)`, `radam(...)`, `rmsprop(...)`, `rprop(...)`, `sgd(...)` directly to use the `foreach` implementation instead of the for-loop for better performance. Applicable means

1. The user has not specified kwargs relating to implementation (`foreach`, `fused`, or `differentiable`),
2. All tensors are native tensors (not subclasses) and on CUDA,
3. `torch.jit.is_scripting` is `False`.

When these conditions are satisfied, the implementation used will match the implementation used when one passes `foreach=True`. The user defined flag for `foreach` will NOT be overwritten in order to preserve user selections. For more details, check the [documentation](https://pytorch.org/docs/stable/optim.html#algorithms). There should be no significant differences between the results returned by these optimizers. To revert to the old behavior, say, for `adam`, pass in `adam(..., foreach=False, ...)` or initialize `Adam` with `Adam(..., foreach=False, ...)`.

Pull Requests: #92306, #92716, #92723,#92724, #92726, #92727, #92728, #92715, #91896, #92730, #90865, #93184, #92181, #92923, #95415, #95818, #95811

### **`torch.nn.utils.stateless.functional_call` now respects tied weights (#90477)**

Assume a module has two tied weights, x and x_tied. Previously, invoking `functional_call(module, parameters_and_buffers, args, kwargs=None, *, strict=False)` with a parameter dictionary of only one of the tied weights would result in the other one(s) not being updated.

We’ve changed the behavior so that providing one of the tied weights in the parameter dictionary will update all other tied weights. If you would like the behavior in previous versions of PyTorch, please set `tie_weights=False`.

Please also see the related deprecation section "torch.nn.stateless.functional_call in favor of torch.func.functional_call".

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
>>> class Foo(nn.Module):
...    def __init__(self):
...        super().__init__()
...        self.x = nn.Parameter(torch.zeros([]))
...        self.x_tied = self.x
...
...    def forward(self, inp):
...        return self.x + self.x_tied

>>> foo = Foo()
>>> params = {'x': torch.ones([])}
>>> result = torch.nn.utils.stateless.functional_call(foo, params, torch.randn([]))
>>> print(result)
1.0
```

</td>
<td>

```Python
>>> class Foo(nn.Module):
...    def __init__(self):
...        super().__init__()
...        self.x = nn.Parameter(torch.zeros([]))
...        self.x_tied = self.x
...
...    def forward(self, inp):
...        return self.x + self.x_tied

>>> foo = Foo()
>>> params = {'x': torch.ones([])}
>>> result = torch.nn.utils.stateless.functional_call(foo, params, torch.randn([]), tie_weights=False)
>>> print(result)
1.0
```

</td>
</tr>
</table>

### **Require `return_complex` to be passed explicitly to `torch.stft` for real input (#86724)**

`torch.stft` takes an optional return_complex parameter that indicates whether the output should be a floating point tensor or a complex tensor. `return_complex` previously defaulted to False for real input tensors. This PR removes the default and makes `return_complex` a required argument for real inputs. However, complex inputs will continue to default to `return_complex=True`.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
>>> a = torch.rand(1024)
>>> _ = torch.stft(a, n_fft=128)
```

</td>
<td>

```Python
>>> t = torch.rand(1024)
>>> _ = torch.stft(t, n_fft=128, return_complex=False)
```

</td>
</tr>
</table>

### **Require inputs to `torch.istft` to be complex valued**

`torch.istft` no longer supports input in the form of real tensors
with shape `(..., 2)` to mimic complex tensors. Instead, convert
inputs to a complex tensor first before calling `torch.istft`.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
>>> t = torch.rand(65, 33, 2)
>>> _ = torch.istft(t, n_fft=128, length=1024)
```

</td>
<td>

```Python
>>> t = torch.rand(65, 33, 2)
>>> _ = torch.istft(t, n_fft=128, length=1024)
RuntimeError: istft requires a complex-valued input tensor matching the output from stft with return_complex=True.
>>> t_complex = torch.view_as_complex(t)
>>> _ = torch.istft(t_complex, n_fft=128, length=1024)
```

</td>
</tr>
</table>

### **Change default behavior of sparse tensor construction to not do component verification(#92094)**

We now disable the costly component verification of torch.sparse_coo/csr/csc/bsr/bsc/compressed_tensor by default. The user can use the new `check_invariants` flag or `torch.sparse.check_sparse_tensor_invariants` to locally enable component verification. This allows users to constrain these costly checks to specific regions of their code and enables better overall performance. Previously users had no access to public constructors that disable these checks.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
>>> i = [[0, 1, 1],
         [2, 0, 5]]
>>> v =  [3, 4, 5]
>>> s = torch.sparse_coo_tensor(i, v, (2, 3))
RuntimeError: size is inconsistent with indices: for dim 1, size is 3 but found index 5
```

</td>
<td>

```Python
>>> i = [[0, 1, 1],
         [2, 0, 5]]
>>> v =  [3, 4, 5]
>>> s = torch.sparse_coo_tensor(i, v, (2, 3), check_invariants=True)
RuntimeError: size is inconsistent with indices: for dim 1, size is 3 but found index 5
>>> with torch.sparse.check_sparse_tensor_invariants():
...     s = torch.sparse_coo_tensor(i, v, (2, 3))
...
RuntimeError: size is inconsistent with indices: for dim 1, size is 3 but found index 5
```

</td>
</tr>
</table>

### **Remove deprecated functionality from `torch.testing`**

Historically, `torch.testing` exposed a lot of private and undocumented functionality publicly. The 2.0 release completes the deprecation cycle for the following items and removes them:

- `rand` and `randn` (#87970)
- `get_all_device_types` (#87971)
- multiple dtype getters (#87972)
- `make_non_contiguous` (#87973)

### **Hooks registered on tensor to always run, even if they are the inputs to `.grad()` (#85849)**

This is a bug fix. Per the docs, hooks registered to Tensor should fire any time gradients are computed w.r.t. to that tensor. This change corrects the behavior to be consistent with the documentation. See [documentation](https://pytorch.org/docs/2.0/notes/autograd.html#backward-hooks-execution) for more details about backward hooks execution..

**2.0**

```Python
a = torch.tensor(1., requires_grad=True)
b = a.clone()
b.register_hook(hook)  # the hook registered here didn't fire before!
torch.autograd.grad(b.clone(), inputs=(b,))
```

### **`grad_fn` post-hooks can always observe the modifications to gradient by any grad_fn pre-hooks or hooks registered to Tensor, even if this is a leaf tensor (#85849)**

This corrects the behavior of hooks to be consistent with the documentation in the case where the tensor is a leaf tensor, i.e. the node is a grad accumulator node. See [documentation](https://pytorch.org/docs/**2.0**/notes/autograd.html#backward-hooks-execution) for more details about backward hooks execution.

**2.0**

```Python
def hook(grad):
   # updates grad
   return grad * 3

def hook2(grad_input, grad_output):
   # Before this change, grad_output would NOT see the x3
   print(grad_output)

a = torch.tensor(1., requires_grad=True)
b = a.clone()
acc_grad = b.grad_fn.next_functions[0][0]
acc_grad.register_hook(hook2)
b.register_hook(hook)
torch.autograd.backward(b.clone(), inputs=(a,))  # hook fire
```

### **Remove FSDP `params_with_grad` (#87480)**

In FSDP, we used to have an API `params_with_grad` for users to get parameters which have gradients from the FSDP module. We decided not to expose this helper because it is not a common paradigm.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
m = FullyShardedDataParallel(module)
m.params_with_grad()
```

</td>
<td>

```Python
m = FullyShardedDataParallel(module)
m.params_with_grad()  # Runtime error thrown
# For work-around, users can still do
[p for p in self.parameters() if p.grad is not None]
```

</td>
</tr>
</table>

### **Users doing wildcard import of torch.distributed.fsdp.fully_sharded_data_parallel will no longer get non-public symbols (#87917)**

Users could previously import both public and non-public symbols:

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
from torch.distributed.fsdp.fully_sharded_data_parallel import *
ShardingStrategy.FULL_SHARD # Non-public API
FullyShardedDataParallel(module) # public API
```

</td>
<td>

```Python
from torch.distributed.fsdp.fully_sharded_data_parallel import *
ShardingStrategy.FULL_SHARD # Non-public API, this will fail now
Fully`Sharded`DataParallel(module) # public API
...
# Users can instead
from torch.distributed.fsdp.fully_sharded_data_parallel import (
FullyShardedDataParallel,
ShardingStrategy,
)
FullyShardedDataParallel(module, sharding_strategy=ShardingStrategy.FULL_SHARD)
```

</td>
</tr>
</table>

### **Signature of FSDP `auto_wrap_policy `related APIs were changed in (#88450).**

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
lambda_auto_wrap_policy(m, unwrapped_params=...)
transformer_auto_wrap_policy(m, unwrapped_params=...)
size_based_auto_wrap_policy(m, unwrapped_params=...)
```

</td>
<td>

```Python
lambda_auto_wrap_policy(m, nonwrapped_numel=...)
transformer_auto_wrap_policy(m, nonwrapped_numel=...)
size_based_auto_wrap_policy(m, nonwrapped_numel=...)
```

</td>
</tr>
</table>

### **Updated `alltoall` signature to be consistent with other c10d APIs (#90569)**

The keyword argument names have been changed.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
alltoall(output=..., input=...)
```

</td>
<td>

```Python
alltoall(output_tensors=..., input_tensors=...)
```

</td>
</tr>
</table>

### **Remove unused functions in torch.ao.quantization.fx.utils (#90025)**

This commit removes the following unused functions from both the torch.quantization and the
torch.ao.quantization namespaces:

- `graph_pretty_str`
- `get_per_tensor_qparams`
- `quantize_node`
- `get_qconv_op`
- `create_qparam_nodes`
- `node_return_type_is_int`
- `is_get_tensor_info_node`

### **Make `torch.ao.quantization.backend_config.BackendConfig` accept inputs in the right order (#90698)**

The existing `BackendConfig` fusion pattern uses a "reversed nested tuple" format that is unintuitive.
This pattern format also complicates the signatures of the user specified "fuser methods", which needed to accept arguments in reverse nested order to match
the patterns:

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
import torch as nn
import torch.ao.nn.intrinsic as nni
from torch.ao.quantization.backend_config import BackendPatternConfig

def fuse_linear_relu(is_qat, relu, bn_conv):
    (bn, conv) = bn_conv
    return nni.ConvBnReLU2d(conv, bn, relu)

config = BackendPatternConfig((nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))) \
    .set_dtype_configs(...) \
    .set_fuser_method(fuse_conv_bn_relu) \
    .set_fused_module(nni.ConvBnReLU2d)

backend_config.configs  # returns Dict[Pattern, BackendPatternConfig]
```

</td>
<td>

```Python
def fuse_linear_relu(is_qat, conv, bn, relu):
    return nni.ConvBnReLU2d(conv, bn, relu)

config = BackendPatternConfig((nn.Conv2d, nn.BatchNorm2d, nn.ReLU)) \
    .set_dtype_configs(...) \
    .set_fuser_method(fuse_conv_bn_relu) \
    .set_fused_module(nni.ConvBnReLU2d)

# Or for backward-compatibility
def fuse_linear_relu(is_qat, relu, bn_conv):
    (bn, conv) = bn_conv
    return nni.ConvBnReLU2d(conv, bn, relu)

config = BackendPatternConfig() \
    ._set_pattern_complex_format((nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))) \
    .set_dtype_configs(...) \
    .set_fuser_method(fuse_conv_bn_relu) \
    .set_fused_module(nni.ConvBnReLU2d)

backend_config.configs  # returns List[BackendPatternConfig]
```

</td>
</tr>
</table>

### **Make the AO codebase compliant with the public vs private API guidelines of pytorch [Public-API-definition-and-documentation](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation)**

If users were using any of the AO private APIs then these would have to be accessed with a preceding `_` to conform with the guidelines.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
get_observer_dict()
```

</td>
<td>

```Python
_get_observer_dict()
```

</td>
</tr>
</table>

Pull Requests: (#86029, #87515, #87516, #87517, #87518, #87519, #88392, #88394, #88396, #88397, #87521, #88395, #87883, #88399, #88398, #86022, #86023, #86024, #86025, #86026, #86027, #86028, #86030, #86031, #86032, #86033, #86034, #86037, #90315, #88391, #90554, #87520)

### **Remove overwrite_output_observer and represent the observer constraints for fixed qparams ops through the existing DTypeWithConstraints mechanism (#88620)**

This commit removes `overwrite_output_observer` and `overwrite_output_fake_quantize` overwrite observer settings in the BackendConfig. Instead, we represent the observer constraints for
fixed qparams ops through the existing DTypeWithConstraints mechanism. Note that, however, to be consistent with other DTypeWithConstraints checks, we no longer throw an error if an incorrect observer is specified, but simply ignore the offending QConfig and log a warning instead. This is the BC-breaking part of the change.
**1.13**

```Python
from torch.ao.quantization.qconfig import default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx

model = ModelWithFixedQParamsOps()
qconfig_mapping = QConfigMapping().set_global(default_qconfig)
example_inputs = ...
prepare_fx(model, qconfig_mapping, example_inputs)
```

Before this commit, running the above leads to an exception because the wrong observers are used for fixed qparams ops. After this commit, the above will only encounter a warning,and the fixed qparams ops will not be quantized. In both cases, switching to `get_default_qconfig_mapping` will cause the fixed qparams ops to be quantized.

### **Remove `torch.ao.quantization.quantization_patterns` and `torch.ao.quantization.fusion_patterns`(#89872)**

The following classes under the `torch.ao.quantization.fx.quantization_patterns` namespace are migrated to the `torch.ao.quantization.fx.quantize_handler`
namespace:

- `QuantizeHandler`
- `BinaryOpQuantizeHandler`
- `CatQuantizeHandler`
- `ConvReluQuantizeHandler`
- `LinearReLUQuantizeHandler`
- `BatchNormQuantizeHandler`
- `EmbeddingQuantizeHandler`
- `RNNDynamicQuantizeHandler`
- `DefaultNodeQuantizeHandler`
- `FixedQParamsOpQuantizeHandler`
- `CopyNodeQuantizeHandler`
- `GeneralTensorShapeOpQuantizeHandler`
- `CustomModuleQuantizeHandler`
- `StandaloneModuleQuantizeHandler`

The following classes under the torch.ao.quantization.fx.fusion_patterns namespace are migrated to the torch.ao.quantization.fx.fuse_handler
namespace:

- `DefaultFuseHandler`
- `FuseHandler`

### **Remove public APIs under the `torch.ao.quantization.fx.backend_config_utils` namespace(#89810)**

The following APIs that were mistakenly public under the `torch.ao.quantization.fx.backend_config_utils` namespace are removed in this commit.

- `get_quantize_handler_cls`
- `get_fusion_pattern_to_fuse_handler_cls`
- `get_native_quant_patterns`
- `get_pattern_to_quantize_handlers`

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
from torch.ao.quantization.fx.backend_config_utils import (
    get_quantize_handler_cls,
    get_fusion_pattern_to_fuse_handler_cls,
    get_native_quant_patterns,
    get_pattern_to_quantize_handlers,
)
all_quant_patterns = get_native_quant_patterns()
```

</td>
<td>

```Python
from torch.ao.quantization.fx.quantization_patterns import (
    _get_quantize_handler_cls,
    _get_pattern_to_quantize_handlers,
)
from torch.ao.quantization.fx.fusion_patterns import (
    _get_fusion_pattern_to_fuse_handler_cls,
)
from torch.ao.quantization.backend_config import (
    get_native_backend_config,
)
all_quant_patterns = _get_pattern_to_quantize_handlers(get_native_backend_config())
```

</td>
</tr>
</table>

### **Update torch.{slice|select|diagonal|as_strided}\_scatter ops to preserve input stride/storage_offset (#91029)**

These operators are primarily used by the [functionalization pass](https://dev-discuss.pytorch.org/t/functionalization-in-pytorch-everything-you-wanted-to-know/965), used in AOTAutograd. Previously, they would always return contiguous tensors. Now, they return a tensor with the same striding as their first argument.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
>>> x = torch.ones(2, 2, 2)
>>> base = x[:, :, 1]
>>> base.stride()
(4, 2)
>>> x = torch.zeros(2, 2, 2)
>>> base = x[:, :, 1]
>>> base.stride()
(4, 2)
>>> torch.diagonal_scatter(base, torch.ones(2)).stride()
# returns a tensor with same strides as base.
(4, 2)
```

</td>
<td>

```Python
>>> x = torch.ones(2, 2, 2)
>>> base = x[:, :, 1]
>>> base.stride()
(4, 2)
>>> x = torch.zeros(2, 2, 2)
>>> base = x[:, :, 1]
>>> base.stride()
(4, 2)
>>> torch.diagonal_scatter(base, torch.ones(2)).stride()
# returns a contiguous tensor
(2, 1)
```

</td>
</tr>
</table>

### **Remove ONNX deprecated monkey patches to torch.Graph (#94747)**

The Deprecated monkey patches to `torch.Graph`, `torch.Block` and `torch.Node` are removed

Monkey patches to the classes `torch.Graph`, `torch.Block` and `torch.Node` from `torch.onnx` have been removed. This means the methods `torch.Graph.op()`, `torch..Graph.at()`, `torch.Block.op()`, `torch.Graph.constant()`, and `torch.Node.__getitem__` are no longer available.

Users creating custom symbolic functions for the `torch.onnx` exporter can continue to assume the `g.op()` interface for creating an operator in the graph, which is now exposed via the `GraphContext` class. Users should not assume any other methods from the `GraphContext` class other than those defined natively by `torch.Graph` and `.op()`.

Code change to existing symbolic functions is not expected with this change.

### **Add full checker mode in torch.onnx.export (#83186)**

This removes boolean value of `full_check` parameter in TORCH API `check_onnx_proto`, and forces `full_check` with warning messages if it fails.

Also, the API didn’t check on types in the graph even with `full_check=True` previously. With the change, a warning message will show if the graph contains type error.

### **C++ API specific BC-Breaking Changes:**

#### **Deleted torch::deploy from PyTorch Core (#85953)**

`torch::deploy` has been migrated to over to [MultiPy](https://github.com/pytorch/multipy). Ongoing development will continue in this repository.

#### **Remove the use of `lazy::View` (#87822)**

The view and aliasing infrastructure in lazy tensor core has been deprecated in favor of functionalization.

#### **Renamed `c10::fromIntArrayRef` to `c10::fromIntArrayRefSlow` and changed call sites (#86235)**

The function has been renamed to more accurately reflect its performance characteristics.

# Deprecations

## torch.func aka functorch

### **We’ve deprecated the functorch module in favor of the new torch.func module**

We’re excited to announce that, as the final step of upstreaming and integrating functorch into PyTorch, the functorch APIs are now available in the torch.func module. Our function transform APIs are identical to before, but we have changed how the interaction with NN modules work.

We’ve deprecated `functorch._` function transforms (e.g. `vmap`, `grad`, `jvp`) in favor of their identical `torch.func._ `counterparts (#92279).
PyTorch has consolidated on `torch.func.functional_call` as the NN module functional API. Please migrate from `functorch.{make_functional, make_functional_with_buffers}` to it. For more details see this [Guide](https://pytorch.org/docs/master/func.migrating.html#functorch-make-functional)
Please migrate from `functorch.combine_state_for_ensemble` to `torch.func.stack_module_state`. For more details see this [Guide](https://pytorch.org/docs/master/func.migrating.html#functorch-combine-state-for-ensemble)
We are no longer supporting functorch.compile (also known as AOTAutograd) as a frontend for compilation in PyTorch; we have integrated AOTAutograd into PyTorch’s compilation story. If you are a user, please use `torch.compile()` instead.

## Python API

### **Deprecate TypedStorage, its derived classes, and all of their public methods (#85303)**

Typed storages have been removed from the C++ side and torch.UntypedStorage is used in place. The use of torch.TypedStorage and all of its subclasses is now deprecated.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
tensor.storage()
torch.TypedStorage(...)
```

</td>
<td>

```Python
tensor.untyped_storage()
torch.UntypedStorage(...)
```

</td>
</tr>
</table>

If you need to access individual elements in a storage as a particular dtype, you can simply create a tensor to view it:

```Python
torch.tensor(storage, dtype=...)
```

### **Deprecate `tensor.mT`,`tensor.T`,`tensor.mH`,`tensor.H` on 0D-tensors (#92143)**

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
>>> a = torch.tensor(10)
>>> a.T
>>> a.H
```

</td>
<td>

```Python
>>> a = torch.tensor(10)
>>> a.T
UserWarning: Tensor.T is deprecated on 0-D tensors. This function is the identity in these cases.
>>> a.H
UserWarning: Tensor.H is deprecated on 0-D tensors. Consider using x.conj().
```

</td>
</tr>
</table>

## Autograd API

### **Deprecate decorating classes with torch.no_grad (#89522)**

Decorating classes with `torch.no_grad` is now deprecated. You should be decorating its functions or methods instead. To preserve the current behavior of class decoration, you can directly decorate the `__init__` method and nothing else.

<table>
<tr>
<th>1.13</th>
<th>2.0</th>
</tr>
<tr>
<td>

```Python
@torch.no_grad()
class Blah():
  pass
```

</td>
<td>

```Python
class Blah():
  @torch.no_grad()
  def __init__(self):
    pass
```

</td>
</tr>
</table>

## Linalg

### **Remove the use of overload at::frobenius_norm(const Tensor&) (#81762)**

In continuation with the deprecation process from release 1.12 the tensor overload for this function has been removed. This function was not used in the bindings of Pytorch and should not impact users of `torch.norm`.

## torch.nn API

### **Canceling deprecation of `functional.{tanh, sigmoid}` functions (#86905)**

Both these ops are heavily used and so will not be removed. Deprecation warnings have been removed.

### **Deprecated torch.nn.utils.stateless.functional_call in favor of torch.func.functional_call (#92280)**

We’ve moved torch.nn.stateless.functional_call under the torch.func module to reflect how it is useful for working with nn.Modules in a functional style. As of PyTorch **2.0**, `torch.func.functional_call` is a drop-in replacement for `torch.nn.stateless.functional_call` and we will remove `torch.nn.utils.stateless.functional_call` in a future version of PyTorch. However, please note that we did change the default behavior of `torch.nn.stateless.functional_call` in PyTorch 2.0 (see “torch.nn.utils.stateless.functional_call now respects tied weights” under BC-breaking notes).

## Releng

### **Deprecated private API torch.\_six (#94709)**

Removed the Python 2 and 3 compatibility library six and future and torch.\_six.
**2.0**

```Python
# from torch._six import string_classes
str
# from torch._six import int_classes
int
# from torch._six import inf, nan
from torch import inf, nan
# torch._six.string_classes
str
```

## Onnx

### **Deprecated Caffe2 ONNX exporter support[ #95071](https://github.com/pytorch/pytorch/pull/95071)**

Users must use PyTorch 1.x versions to use Caffe2 ONNX exporter. This capability will be completely removed from PyTorch 2.x series.

# New features

## torch.nn API

- Add `torch.nn.functional.scaled_dot_product_attention()` to allow writing fast Transformer-like functions and use it to speed up `nn.Transformer()` ( #91362, #91066, #90413, #87312, #94008, #89470, #90776, #92189)
- Add hooks for `Module.register_{buffer,module,parameter}` functions (#86148, #87369)
- Add `Module.full_backward_pre_hook` (#86700)
- Add `Module.state_dict_pre_hook` (#90435)
- Add `Module.call_super_init: bool` flag that can be used to ensure `Module` initialization is properly calling parent’s `__init__` (#91819)

## torch.func

- Add `functorch` support [for torch.autograd.Function](https://pytorch.org/docs/master/notes/extending.func.html): one is now able to apply function transformations (e.g. vmap, grad, jvp) over torch.autograd.Function. (#92023, #91452, #91222, #90037, #90077, #90966, #89860, #91211, #92030)
- Add support for linearize a-la [jax.linearize](https://jax.readthedocs.io/en/latest/_autosummary/jax.linearize.html#jax.linearize) (#94173)
- Add torch.func.functional_call, a new utility function to work with NN modules. (#89213)
- Add torch.func.stack_module_state, a new utility function to help with model ensembling (#88850)

## Cuda

- Introduce CUDA Device Assertions Infrastructure (#84609)
- `Logcumsumexp` for complex dtypes for CUDA (build-time optimized) (#94310)
- Caching allocator tracing (#86241)
- Add Pluggable CUDA allocator backend (#86786)
- Add cudaMallocAsync as an alternative backend for the CUDA allocator (#82682)

## Cpp API

- Add `set_to_none` flag for C++ optim endpoint (#92989)

## NestedTensor API

- Add support for `tensor.to()` for NestedTensor backend (#87146)
- Add backwards support for `gelu` and `relu` operators (#94776)
- Add support for `torch.neg` operator (#88131)

## Distributed

- Distributed Tensor (Prototype Release)
  - PyTorch [DistributedTensor](https://github.com/pytorch/pytorch/blob/master/torch/distributed/_tensor/README.md) (DTensor) is a prototyping effort with distributed tensor primitives to allow easier distributed computation authoring in the SPMD (Single Program Multiple Devices) paradigm. The primitives are simple but powerful when used to express tensor distributions with both sharded and replicated parallelism strategies. PyTorch DTensor empowered PyTorch [Tensor Parallelism](https://pytorch.org/docs/master/distributed.tensor.parallel.html) along with other advanced parallelism explorations. In addition, it also offers a uniform way to save/load state_dict for distributed checkpointing purposes, even when there’re complex tensor distribution strategies such as combining tensor parallelism with parameter sharding in FSDP. (#88176, #88177, #88178, #88179, #88551, #88549, #88550, #89800, #89967, #89968, #89991, #90106, #90241, #90449, #90731, #90732, #90733, #90734, #90735, #91756, #91783, #91785, #91801, #91802, #92069, #92197, #92198, #92290, #92611, #92651, #92652, #92677, #93040, #93160, #93306, #93832, #93957, #94517, #94524)
  - We also design and implement Tensor Parallel & 2D Parallel (Tensor Parallel + FSDP) on top of DistributedTensor. (#88180, #89242, #89467, #89535, #89779, #89878, #93029, #93412, #94421, #94748)
- Distributed Checkpoint
  - PyTorch Distributed Checkpointing (DCP) API was first introduced in PyTorch 1.13 and this will be an official prototype release in PyTorch 2.0. The distributed checkpoint API in PT2.0 decouples the storage layer from the checkpoint planning layer. Planner types are introduced to perform the coordination of storage both locally and globally to plan the save/load process. Checkpointing support for FSDP `sharded_state_dict` is added as well. (#87987, #88698, #89256, #89398, #89399, #89501, #89503, #89537, #89542, #89873, #89964, #90212, #91036, #91092, #91209, #91269, #92553, #92705, #92829, #92869, #92933, #94379, #94501)
- DistributedDataParallel
  - Enable DDP for PyTorch 2.0 (#87549, #88523, #89096, #88460, #88480, #88521, #94749, #93162, #89802, #92986)
- FullyShardedDataParallel
  - Add the option to use the original parameters via `use_orig_params=True` in the FSDP constructor (#84911)
  - Enable the use of TorchDispatch with FSDP (#88014)
  - Hybrid Sharded Data Parallel (#89915)
  - Enable FSDP for PyTorch 2.0 (#88781, #89330, #89523)
- Distributed (c10d)
  - Dispatchable collectives: An improvement to the existing `init_process_group` API which changes backend to an optional argument. For users, this feature will allow for code that runs on both GPU and CPU machines without having to change the backend specification. The dispatchability feature will also allow users to perform both GPU and CPU collectives using the same ProcessGroup, as PyTorch will automatically find an appropriate backend for the tensor type (as of PyTorch 2.0, the default is NCCL for CUDA and Gloo for CPU). Existing backend specifications by users will be honored and will not require change (#83679, #83735, #83810, #83859, #83876, #83916, #84423, #86166, #86368, #86407, #86408, #86409, #88351, #88846, #88889, #88903, #89317, #89318, #89505, #89813, #88330, #91257, #91172)

## Mps

- Add native support for:`torch.nn.functional.group_norm`(#91190), `torch.var_mean` (#91190), `torch.nansym`(#93845), `torch.frac`(#86625), `torch.signbit`(#87214), `torch.exp1m`(#87147), `torch.cumsum`(#88319), `torch.trace`(#87910), `torch.nn.Hardswish` (#87952),`torch.inverse`(#90428), `torch.floor_divide`(#91126), `unfold`(#91266), `bincount`(#91267), `nonzero`(#91616), `norm_dtype`and`cdist`(#91643), `unique`and`unique_consecutive`(#88532), `nan_to_num`(#91110), `torch.linalg.cross`(#91642), `randperm`(#91708), `triangular_solve`(#94345), `grid_sampler2d`(#94273), `remainder`(#92139), `addr`(#94538), `fmod`(#94722), `repeat_interleave` (#88649),`sort`and`argSort`(#94697),`range` (#91075)
- Add functions to handle rng and force device synchronization `torch.mps.{get_rng_state, set_rng_state, synchronize, manual_seed, seed}` (#94417)
- Add support for the `mps` device for `torch.Generator` (#91348)
- Add `torch.int64` support for unary ops (#86615)

## Profiler

- Improve Memory Profiler(alpha): enhancement to the existing memory profiler that can attribute memory consumptions to activations, gradients, parameters, and optimizer states (#86802, #86853, #86854, #86880, #87006, #87566, #87567, #87568, #88924, #88925, #88926, #89355, #89356, #86355, #88917, #87133, #86753, #86754, #87096, #86909, #87825)
- Add Linux perf event support in profiler (#87866, #87874)

## Foreach API

- Implement:
  - `torch._foreach_lerp` (#87562),
  - `fused adamw` (#88015)
  - `_foreach_addc`(div/mul)(\_).Tensor (#88157)
  - `clamp_min` `clamp_max` (#91384)
  - `adamw` (#88015)

## Mobile

- Add XNNPACK Delegate Framework.
  - Enable a XNNPACK graph to be built from the torchscript IR and performing checks (#86980, #87128, #87824)
  - Add flatbuffer serialization support (#87826, #87906, #87907, #87908)
  - Create `Executor` and `Compiler` classes which compiles the XNNPACK graph and preps for execution (#88779, #88778, #88780, #89090)
  - Optimize library includes (#88863, #89231)
  - Add Constant Data which will be used in Convolution (#89445)
- Add support for better benchmarking
  - Add support in lite_predictor benchmark binary to select event lists and perform benchmarking using Linux perf through Kineto profiler (#87876)
  - List all missing ops at once (#94205)

## Sparse API

- Add `torch.sparse.check_sparse_tensor_invariants` context manager that allows users to opt into more checks at runtime for better debugging. (#92094)
- Add `check_invariants` flag to `torch.sparse_coo/csr/csc/bsr/bsc/compressed_tensor `to allow users to verify components at construction time. (#92094)
- Add `reduce` flag for CPU to torch.sparse.mm with support for `sum, mean, amax, amin` (#83727)

## Optimizer API

- Make `{Adadelta, Adagrad, Adamax, AdamW, ASGD, NAdam, RAdam, RProp}` differentiable (#86096, #86258, #86183)
- Publicly expose \_LRScheduler to LRScheduler (#88503)

## Distributions

- Add a transform for positive-definite matrices. (#76777)

## Signals

- Set up new module torch.signal.windows (#85599)
- Add the Nuttall window to signals/ (#90103)
- Implement old singal/windows in Python (#87082, #87330)

## Quantization

- Add support for oneDNN backend for server CPU quantized inference (#91056, #88478, #88665, #88668, #88879, #88923, #89188, #91297, #90262, #90364, #91152, #91153, #91154, #91155, #91934, #88661)
- Add new ‘x86’ backend to be used for quantized CPU inference (#91235, #88799)

## Vulkan

- Add Vulkan support for several torch operators:
  - `torch.abs` (#87414)
  - `torch.select` for height and width dimensions (#94612)
- Vulkan optimization passes now automatically apply data transfers between the CPU and GPU for input and output tensors (#87432)
  - If the `requires_backend_transfers` flag of a model is set to `false`, then input tensors do not to be transferred to the GPU (via `tensor.gpu()`) and output tensors do not to be transferred back to the CPU (via `tensor.cpu()`) since these transfers are inserted into the model
  - To avoid inserting data transfers into a model, add `MobileOptimizer.VULKAN_AUTOMATIC_GPU_TRANSFER` under `torch.utils.mobile_optimizer` to the `optimization_blocklist` argument of `optimize_for_mobile` (#92081)

## ROCm

- `hipGraph` support for pytorch mainline (#88202)

## Fx

- Introduce symbolic shape guards (#87570, #90528, #90665, #90679, #90876, #91058, #93894, #94782)
- Introduce a match filter for SubgraphRewriter (#86430, #87998, #87257)
- Support list-typed args in PatternMatcher (#88656)
- Add `any_chain()` in operator support (#90949)
- Have replace_pattern return replaced nodes (#90244)

## Jit

- Allow freezing JIT modules that contain mutable interfaces (#86039, #91020)
- ApplyLinear-BatchNormNd folding during torch.jit.freeze (#86706)
- Add an option to skip loading of debug traces, in order to reduce memory usage (#91430)
- Introduce torch.jit.\_drop function modifier to avoid compiling a method on a non-nn.Module class (#93012)
- Allow providing a kwargs-like dict of example inputs to torch.jit.trace with the new `example_kwarg_inputs` argument (#81623, #94032)
- Include example input shapes when serializing jit.traced modules to assist with debugging (#90744)

## Build

- Add Ada Lovelace (cuda arch sm8.9) support (#87436)
- Add an option to disable TORCH_WARN and TORCH_WARN_ONCE log (#87188)
- Enable memory map file support for Android, Apple, and CXX (#88545)
- Support DNNL_GRAPH_CPU_RUNTIME=TBB build option (#87512)

## ONNX

- Verification tool to find mismatch in model export (#89946,[ #89807](https://github.com/pytorch/pytorch/pull/89807),[ #89808](https://github.com/pytorch/pytorch/pull/89808),[ #89947](https://github.com/pytorch/pytorch/pull/89947),[ #94648](https://github.com/pytorch/pytorch/pull/94648))

## Cudnn

- Add an environment variable to skip cudnn version compatibility check (#89184)
- Enable cuDNN Frontend v8 API by Default (#91117)

# Improvements

## Python API

- Set std/var correction overloads default value to None (#56398)
- Implement correction argument in torch.masked.{std,var} (#87118)
- Update `torch.squeeze` to allow squeezing multiple dimensions at once (#89017)
- Add support for int32 indices in index/index_put ops (#86309)
- Enable `where` to have cpu scalar args (#87022)
- Add support for NumPy scalars to `torch.tensor.asarray` (#90914)
- Update opt_einsum to have more reasonable defaults (#86985)
- Improve error message for `Tensor.set_` when dtypes mismatch(#88804)
- Enable out variant of `torch.max`(#85926)
- Implement faster gradient clipping using foreach function (#91846)

## Autograd API

- Add backward support for `torch.ormqr` (#86800)
- Pre-hooks registered on tensor are guaranteed to run before pre-hooks registered on grad_fn (#85849)
- Add a new overridable method `setup_context` (#89859, #92312)
  - You must use override this method if you plan to use your autograd Function with functorch
  - If you choose to override this method, `forward` should no longer take ctx as an input.
- Add context manager `torch.autograd.set_multithreading_enabled` for disabling multithreading in the autograd engine (#86245)
- Add backward AD support for unary foreach functions (#89591)

## torch.nn API

- Add `remove_duplicate` flag to `Module.named_buffers()` method (#84984) and `Module.named_parameters()` (#88090)
- Add kwarg support for `Module` forward-pre and forward hooks (#89389)
- Improve error message for `Transformer()` fast path (#90783) and kernel selection (#90783)
- Add support for `torch.bf16` for `Embedding` (#94163)
- Add `freeze` argument to `Embedding()` (#86769)
- Add `torch.channels_last_3d` support for `SyncBatchNorm()` (#88401)
- Add `torch.bfloat16` support on CPU for `functional.{mish,hardtanh,silu}` (#82460)
- Add support for inputs with different data types for `LayerNorm()` (#81851, #88064), `BatchNorm{1,2,3}d()` (#84410), `GroupNorm()` (#89485, #81852, #88663, #92671, #92668)
- Improve printing of `ModuleList()` (#90452)
- Add `torch.uint8` support for `functional.interpolate()` on CPU (#90771)
- Make `functional.max_pool1d` error checking consistent between CPU and CUDA (#90211)
- Add `SyncBatchNorm()` fallback to `BatchNorm()` when it is used in a non-distributed setting (#89706)
- Add channels-last support for `GroupNorm()` on XPU (#87680)
- Add `is_causal` kwarg to `TransformerEncoder()` layer (#90508)
- Add `prepend` argument to `Module` hooks to register a hook that will be called before the existing ones (#87370)

## Distributed

- Activation checkpointing
  - Return `None` from `apply_activation_checkpointing` (#87871)
  - Enable non-reentrant support for `checkpoint_sequential` (#86331)
  - Separate CPU offload activation to its own wrapper (#85459)
- DistributedDataParallel
  - Add `PackedSequence` support when `device_ids` is specified (#86614)
  - Enable DDP to handle custom dataclass forward outputs (#92334)
- Distributed (c10d)
  - Add sequence number support for UCC PG (#85047)
- FullyShardedDataParallel
  - Default to `BACKWARD_PRE` for the backward_prefetch of FSDP (#88428)
  - Skip collective communications for `NO_SHARD` in `clip_grad_norm_` (#89137)
  - Allow handle training state to be both `BACKWARD_PRE` and `BACKWARD_POST` in the post-backward assert (#89791)
  - Limit all gather after pre-unshard (#89057)
  - Include module classes in `ModuleWrapPolicy.__repr__` (#89058)
  - Apply the "largest" dtype across all parameters/gradients as defined by PyTorch's type promotion semantics for the total norm returned in `clip_grad_norm_` for low prec grads (#90028)
  - Introduce `ModuleWrapPolicy` for simplicity in FSDP autowrap (#88450)
  - Enable mixed hybrid/non-hybrid sharding strategies (#90846)
  - Re-support model dtype change after FSDP init (#91192)
  - Enable `use_orig_params=True`, `no_sync` and mixed precision to work together (#91193)
  - Enable `summon_full_params(with_grads=True)` (#85738, #87314)
  - Add `keep_low_precision_grads` support when CPU offloading (#86495)
  - Consolidate FSDP `state_dict` `offload_to_cpu` settings (#86211)
  - Add `set_state_dict_type` API to setup `state_dict_type` without using context manager (#86243)
  - Enable the support of `use_orig_param` for FSDP’s `optim_state_dict` (#89898, #89899, #89900)
  - Enable nested FSDP wrapper to use different mixed precision (#90523)
  - Enable input cast skip in MixedPrecision (#90620)
  - Publish `optim_state_dict` and `optim_state_dict_to_load` for FSDP (#90798, #91343, #92744, #92118, #92991, #92992, #93285, #93318, #94109, #94129)
  - Make default input casting in root module only and enable the ability to set different mixed precisions for different submodules (#91365)
- Torch Elastic
  - Update `torchrun` and `TorchElastic` to take optional `local_addr` param to allow skip local IP lookup if specified (#88922)

## torch.func

- Update vmap to accept None(s) in out_dim (#91644)
- torch.func.jacrev: Support chunked computation (#89376, #91326)
- vmap: chunk_size support (#91157)
- torch.vmap: Implement checks (rather than internal asserts) for vmap escaped errors (#89585)
- Avoid calling allclose in the backward if there are tensor subclasses (#91444)
- Refactor NN stateless APIs by swapping module tensors (#92536)

## Cuda

- Use binary units for CUDA memory summary (#91854)
- Improve perf by avoiding implicit string creation in c10_cuda_check_implementation (#88350)
- Add option to record C++ backtraces in \_record_memory_history (#86145)
- Set CUDA_MODULE_LOADING to LAZY when not set by the user (#85692)
- Add warning if captured graph is empty (#88754)
- Add option to dump a captured graph for debugging (#85519)
- Add support to foreach torch zero for bfloat16s (#90437)
- Enable bfloat16 for hardtanh_backward_cuda (#91511)
- Use pytree to allow any input format for cuda graph (#90941)
- Add requested_bytes to CUDA Caching Allocator Stats (#88575)
- Add an option to disable reduced precision reductions for BF16 GEMM (#89172)
- Add an env variable to disable addmm_cuda_lt kernel (#91436)

## Serialization

- Add XPU backend to support torch.save and torch.load (#89679)

## Cpp API

- Reduce ambiguity in Tensor namespace collisions (#92266)

## Dataloader API

- Add support for pin memory on xpu device (#86545)
- Add type annotation to `get_worker_info` (#87017)
- Allow prefetch factor to be optional (#88972)

## NestedTensor API

- Add add/mul for nested dense [B, *, D], [B, 1, D] case (CUDA-only) (#88289)
- Add support for torch.select over irregular dimensions (#88585)
- Add torch.nested.nested_tensor() constructor (#88213)

## Complex API

- Improve complex support for: `torch.nn.functional.conv_transpose3d `(#87967), `torch.log1p` (#89214,#90422), `torch.lerp` (#75584), `torch.logcumsumexp` for CPU (#93153)
- Solve under/overflow for complex division (#92539)

## Composability

- Improve coverage of primtorch and torch.\_ref decompositions: `prims.clone` (#86705), `ndtr, ndtri, log_ndtr, erfcx` (#86077), `NLL loss` (#81128), `conv backward` (#87047), `xlogy and xlog1py` (#77712), `alpha_dropout` (#87989)
- More operations now work with meta tensors: `_adaptive_avg_pool2d_backward` (#86359), (#87074), `avg_pool2d and avg_pool2d_backward` (#87043), `scalar_tensor and argmax` (#88590), `topk` (#88694), `max_pool2d_with_indices_backward` (#88743), `grid_sampler_2d_backward` (#88745), `linalg_cholesky` and `linalg_cholesky_ex` (#89430), `aten._cdist_forward` (#90042), `aten.pixel_shuffle` (#91605)

## Linalg API

- Fix typos in messages under aten (#88964)

## Mobile

- Improve CoreML logging and dependent libraries.
  - Updated Cocoapods (#88075)
  - Preserved CoreML errors by using special throw macro when encountering CoreML API errors (#86938)
- Clean Up MobileOptimizerType Rewrite Flags Public API and Documentation (#91600)
- Clean up flatbuffer lib dependency and fixed its test to match pkl models (#86041, #93022)
- Type corrections to avoid unnecessary static_casts (#93898)
- Add flake8-logging-format linter (#90805, #94840)

## Sparse API

- Add autograd support for `linear` (#86137, #86302), `mm`, `log1p`(#86301, #88155), `to_sparse_*`(#90281)
- Improve support for `sparse_dim`, `dense_dim` (#86203, #86203), `torch.sum`(#86300, #92979), torch.sparse.sampled_addmm`(#86401),`frac`, `deg2rad`, `rad2deg`, `relu`(#88153, #88156, #88442, #86749),`conj()`(#91695),`to_sparse`(#90718),`sparse_mask` (#92248, #94829)
- Add support for per batch index contiguity in CSR/CSC/BSR/BSC (#91243), non-contiguous values in CSR/CSC/BSR/BSC (#91243), non-zero dense_dim to COO/CSC/BSR/BSC/Strided conversions. (#90177), uncoalesced operands to `sparse_mask` (#91964)
- Improve error messages for `indices, values, (c)row_indices, (c)col_indices` (#93149) and `addmm` (#94843)
- Extend gradcheck to BSR and BSC inputs. (#90719)
- Sort BSR indices as part of CSR to BSR conversion (#90918)

## Cpu

- Implement aten::native_batch_norm.out for CPU (#88604)
- Log1p for complex in CPU (#89691)
- Enable oneDNN implementation for LSTM (#91158)

## Package

- Add better debugging for torch.package (#92939)

## Quantization

- Remove weight arg from DTypeConfig for non-weighted ops (#86335)
- Add get_symmetric_qnnpack_qconfig_mapping for XNNPACK quantized ops (#87002)
- Add assert for backend correctness in get_default_qconfig related apis (#86259)
- Replacing List[QConfigMapping] in parallel numeric profiler (#86922)
- Check the fixedqparam op qconfig based on backend_config (#87425)
- Explicitly set default quantized engine instead of relying on the order of supported_qengines (#89804)
- Support setting qconfig by module_type in QConfigMapping in PT 2.0 export flow (#92355)
- Migration of quantization code from torch._ to torch.ao._ (#86171, #86172)
- Improvements to qnnpack fully connected sparse ops (#85243, #85244, #85245, #85246, #85247)
- Support lowering of channel shuffle in FX (#83731)
- Remove explicitly default QConfigMapping settings (#90066)
- quant: make various configs printable (#91419)
- Enable FX quant for patterns like x.view(x.size(...), ...) (#90001)
- X86 qengine always uses fbgemm kernels on OS other than Linux (#93218)
- Change prepare_fx and convert_fx to preserve the GraphModule type of input (#94412)
- update xnnpack to newer version and update API usage in pytorch (#94330)
- Remove \_input_output_observed from backend_config (#92589)
- Add support for LSTM Structured Pruning prune_functions + pattern (#90801)
- Enable FX static quantization for LSTM (#85068)
- Allow setting fixed quantization params for inner LSTM ops (#88456)
- Add support for GRU in fx graph mode quantization (#91976)

## ONNX

- Operator support `col2im` opset 18 (#84594), `mse_loss` (#90717), `aten::contains` (#91660), src/index dynamic axes support for `aten::scatter_add` (#90090), `aten::zero` (#91731), Raise Unsupported for `GridSample` with volumetric 5D input (#92212)
- Pretty print diagnostic logging (#88261)
- Bump onnx to 1.13.1, onnxruntime to 1.14.0 (#90332,[ #94767](https://github.com/pytorch/pytorch/pull/94767))
- Add full graph checker option for `torch.onnx.export` API (#83186)
- Integrate all ONNX operators with a new `JitScalarType` API (#87245)
- Add `share_from_this` to `torch::jit::Graph` (#87343)
- Use optional op to keep None in results for ONNX internal tests (#84789)
- Add support for autograd function inlining in `ONNX_ATEN_FALLBACK` mode (#85736)
- Default runtime type checking to raising errors (#86555)
- Remove the `INT64_MAX` magic numbers (#88341)

## Fx

- Refactor graph partition to check for cyclic dependency (#86511)
- Enable nvprims.transpose fusions for nvFuser (#86967)
- Simplify magic method definition code. (#88017)
- Add sym_floor, sym_sqrt, sym_int (#88760)
- Propagate .meta info when replacing subgraphs in fx (#87255)
- Make `torch.fx` compatible with Python-3.11 (#92895)
- Add type(module) to be stored in the module stack (#87149)
- Ensure that symbolic variables incorporate fresh constraints before they're used (#87254)
- Add type annotation to `getitem` node before `split_module` (#88510)
- Implement pass for annotating getitem nodes (#90237)
- Guard Symbol and ShapeGuardPrinter behind HAS_SYMPY (#90704)
- Copy meta field in fx.GraphModule on deepcopy (#92062, #92623)
- Match get_attr when comparing nodes (#91657)
- Make **deepcopy** of fx.GraphModule handle circular reference. (#93038)
- Populate memo in deepcopy BEFORE copying children. (#93295)

## Mps

- Add fp16 support for `torch.nn.Linear` (#89774), `torch.nn.GELU` (#86218)
- Add support for empty Tensors in `torch.bitwise_not` (#87286), `torch.nn.LayerNorm` (#94212), many backward functions (#94343), `torch.nn.functional.hardswish` (#94342), `torch.topk` (#91884), `torch.arange` (#94485), `torch.linal.inv` (#94551),
- Improve error message for `nn.Conv2d` when inputs are on different devices (#86303)
- Add support via fallback for `torch.nn.{Fold, UnFold}` (#94491)
- Add support for reduction ops on multiple axis at a time (#91734)
- Add support for `k` greater than 16 for `torch.topk` (#94639)

## Build

- Add @pytorch in tools/bazel.bzl (#91424)
- Change visibility for //c10:headers (#91422)
- Simplify OpenMP detection in CMake (#91576)
- Use `@pytorch//` in bazel build files which improves embedding usecases (#89660)
- Enable `USE_CUDA `for bazel build (#92640)
- Add missing default initializers to class members (#94049)

## Jit

- Skip builtins while enumerating class methods (#91805)
- Support lovelace for NVRTC (#87611)
- Expanded symbolic shape support (movedim) (#91696)

## Releng

- Update CI test environment; Add symbolic functions (#94564)
- Import `Literal`, `Protocol`, and `Final` from standard library `typing` as of Python 3.8+ (#94490)
- Add cpuinfo to collect_env.py for new issues reporting which helps triaging on CPU (#93899)
- Refactor nvfuser build (#89621)
- Add error checking to flaky test bot platform parser (#86632)
- Make LazyGraphExecutor extensible (#87218)
- Delete BUILD_SPLIT_CUDA option (#87502)
- Use faster cache flush in triton benchmarking (#88557)
- Guard global observer init against Edge profiler (#86347)

# Bug fixes

## Python API

- Fix as_strided_scatter derivative formula(#87646)
- Add bfloat16 support to torch.prod (#87205)
- Disable dimension wrapping for scalar tensors (#89234)
- Fix SIGSEGV on a big-endian machine when reading pickle data (#92810)
- Fix BC-breaking change to reduction arguments `amin`/`amax` (#93091)
- Fix incorrect tensor storage check (#86845)
- Ensure einsum contracts left to right (#87199)
- Add nondeterministic error for `torch.tensor.scatter` (#88244)
- Fix multi-index for `torch.tensor.index_select` over scalar tensor (#94347)
- Add scalar support for `torch.tensor.where` (#92849)
- Improve error message for unsupported argument types (#87601)
- Change as_strided_scatter’s storage offset default to None from 0 (#87481)
- Make `torch.histc` consistent between CPU and CUDA (#87832)
- Add float to list of allowed ops for serialization (#94910)
- Fix numpy1.24 deprecations in unittests ([#93997] (https://github.com/pytorch/pytorch/pull/93997))
- Properly moving segment_reduce to be private as expected (#93166)

## Autograd API

- Fix behavior of hooks registered to Tensors that had previously been modified in-place (#92734)
  - Previously hooks registered to a tensor after it is modified in-place would erroneously receive the gradients of the output w.r.t. to that tensor before it is modified in-place if that tensor had previously had a hook registered to it before it was modified in-place.
  - See [documentation](https://pytorch.org/docs/2.0/notes/autograd.html#behavior-of-tensor-hooks-when-tensor-is-modified-in-place) for more details about backward hooks execution when tensors are modified in-place.
- Update saved variable hooks to no longer trigger on wrapped numbers (#87316)
- Modifying a view created in no-grad mode in-place no longer triggers an internal assert (#88243)
- Improve error message when saved tensor is detached inplace (#88860)
- Prevent module full_backward_hook from erroring in double backward (#88357)
- Fix forward AD custom Function non-differentiable outputs (#90787)
- Don't materialize forward grad for non-differentiable types (#91183)
- Return input as-is if marked dirty even when requires_grad=False (#91214)
- Fix saved tensor hooks to propogate errors back to python as-is (#94456)
- Fix NumPy broadcasting for backward of `linalg.solve` (#91456), `linalg.lstsq` (#91460)
- Fix torch.var backward when input numel == correction (#94546)
- Fix CopySlices logic to ensure wrapped node runs properly. (#89812)

## torch.nn API

- Fix for RNN-like `Module`s to work with `stateless.functional_call()` (#91111), better error messages (#87442),
- Add missing dim checks `EmbeddingBag` (#85433)
- Fix `Upsample` and `EmbeddingBag` module printing (#93850)
- Fix segfaul in `Conv3D` CPU implementation (#94325)
- Fix overflow issue in `Upsample` (#94290)
- Fix `functiona.pixel_{shuffle,unshuffle}` to consistently return views or not (#86608)
- Fix 64bit indexing `Conv3d()` (#87527), `Upsample()` (#87901)
- Fix preserving requires_grad-ness in fusion utils (#89100)
- Fix support for empty inputs/outputs for `Conv{1,2,3}d()` (#86521), `functional.adaptive_{avg, max}_pool()` (#88906)
- Fix buffer overflow in `Upsample()` (#89252), `MaxUnpool3d()` (#94372)
- Fix `functional.grid_sample()` loss of precision for `torch.float16` inputs (#90427)
- Fix `functional.interpolate()` bicubic interpolation to properly preserve memory format (#90470)

## torch.func

- Fix cross to match unbatched behavior (#86926)
- Properly error on complex inputs or outputs in jacrev, jacfwd (#94805)
- Fix batching rule for dropout (#92975)
- Fix vmap and anomaly mode interaction (#92672)
- Fix and update type hints for `make_functional.py` (#91579)
- torch.tril & torch.tril : add out of bound checks (#89384)
- Fix torch.cat batching rule (#86932)
- Fix reduction boxed batching rules (#91109)

## Cuda

- Check SM version before calling flash attention with BFloat16 (#86600)
- Add range check to multi margin loss target (#89008)
- Fix NVML visible device parsing (#92315)
- Take `CUDA_VISIBLE_DEVICES` into account for nvml calls (#94568)
- Fix topk IMA (#93095)
- Fix: half reduction with multiple sub-iterators (#85596)
- Fix segfault when swapping custom allocator (#89613)
- Conditionally set device in autograd engine (#91191)
- Store `autocast_gpu_dtype` in `custom_fwd` and `custom_bwd` for BFloat16 autocast (#88029)
- Do not use at::cuda::getDefaultCUDAStream() (#91180)
- Ensure that our error handling runs with the GIL enabled (#92848)
- Fix C10_CUDA_CHECK for failing to capture last cuda error occasionally (#93192)
- Fixes a memory leak by making autocast cache global instead of thread-local (#86492)
- Take `CUDA_VISIBLE_DEVICES` into account for nvml calls (#94568)
- Explicitly set the workspace for cuBLAS handles (#86645)

## Cpp API

- Fix CUDNN_PATH handling on Windows (#88898)
- Fix typos in warning/error messages(#88961)
- Remove uneeded checks from embedding bag impl (#92982)
- Fix c++ : segfault in modulelist and moduledict (#93074)

## Visualization

- Fix overflow issue in tensorboard image summary (#90423)
- Remove deprecated call to tf.io.gfile.get_filesystem (#89832)

## NestedTensor API

- Enable non-contiguous Nested Tensors for BMM inputs for NT on CUDA (#88108), linear backward (#94317)
- Fix bug in unsqueeze_nested stride calculation (#88688)

## Distributed

- Distributed(c10d)
  - Fix a static initialization order fiasco in c10d (#90149)
  - Fix `send`, `recv` return type (#92152)
  - Fix MPI backend PG initialization (#92847)
  - Fix header-filter for clang-tidy c10 and apply some fixes to c10 and c10d (#91178)
  - Fix `backend_type` for backend/PG plugin (#93129)
  - Fix UCC PG barrier (#86961)
  - Properly finalize unsuccessful UCC collective posts (#89306)
  - Add pre & post processing for UCC CPU collectives (#89030)
  - Re-enabl `isinstance` with `torch.distributed.ReduceOp` (#87303, #88275)
  - Ameliorate custom `__eq__` for `ReduceOp` (#90088)
  - Fix warning if backend registers timer (#91702)
- DistributedDataParallel
  - Fix DDP when the number of output features is zero (#87793)
- FullyShardedDataParallel
  - Fix `use_orig_params=True` for reentrant activation checkpointing by disabling the post-backward hooks (#87413)
  - Re-establish the wrapped module in `_lazy_init` in case module changing after FSDP constructor (#87837)
  - Fix the incorrect norm calculation for `NO_SHARD` by handling sharded and non-sharded parameters differently in `FSDP.clip_grad_norm_` (#88955)
  - Pass through `ActivationWrapper` directly to the inner wrapped module to fix `state_dict` issues (#87950)
  - Remove the clean of FQNs even for `use_orig_params=True` in FSDP (#91767, #92662)
  - Restrict meta model check to non ignored modules in FSDP (#86766)
  - Fix `keep_low_precision_grads=True` for `use_orig_params=True` (#90027)
  - Fix for `use_orig_params=True` + `no_sync` (#90546)
  - Fix `no_sync`, `use_orig_params=True`, mixed precision, sharded (#92874)
  - Fix input grad propagation when using param mixed precision (#90921)
  - Fix `_mp_shard` in `record_stream` (#91096)
  - Fix "use-after-free" in reshard logic (#94859)
  - Fix `clip_grad_norm_` issues (#94835), (#86337)
  - Fix `load_sharded_state_dict` FQN mismatches for shared parameters (#86524)
  - Fix grad zero vs. `None` edge case (#87308)
  - Fix FSDP `state_dict` transformations of modules with persistent buffers failure with mixed precision enabled (#93396)
  - [FSDP] Fix `nn.Parameter` usage for 2D and `use_orig_params=True` (#89782, #89845, #90562)
- RPC
  - FFixixed use after free in tensorpipe agent (#87627)
- Torch Elastic
  - Make TorchElastic timer importable on Windows (#88522)
- Tensor parallel & 2D parallel
  - Fix the logic to trigger load hooks for 2D parallel integration with FSDP. (#86272)

## Profiler

- Minor bug fixes for ROCM tracing (#89785, #88207)

## Foreach API

- Fix `_foreach_norm` on some tensor sizes (#91844)
- Exempt `_foreach_norm` from autograd_not_implemented_fallback check (#93995)

## Complex API

- Fix serialization of `conj` and `neg_view` (#88182)

## Linalg API

- Add empty tensor check to \_compute_linear_combination (#94245)

## Optimizer API

- Fix discrepancy between mt vs st impl (#92699)
- Do NOT inplace modify gradients (#92706)
- Fix memory leak in \_LRScheduler.step() (#85602)
- Look up `group["capturable"]`, not `defaults["capturable"]` in Adam(W) (#94149)
- `FusedAdam(W)` should take `OptState` into account before unscaling grads (#94060)
- Fix LinearLR scheduler start_factor (#86695)
- Keep AveragedModel buffers in sync when use_buffers=False (#84054)
- Fix OneCycleLR error log (#92040)
- Fix SparseAdam consuming iterator (#86210)
- Fix empty grad support for SparseAdam (#86459)

## Serialization

- Fix set pickle_module if not specified (#88570)
- Explicitly check filelike arg of `torch.save` (#88867)
- Fix dtype mismatch for unallocated storage deserialization (#91285)
- Add float to list of allowed ops (#94910)

## Composability

- Fix segfault in has_torch_function (#88559)
- Fix for usages of **torch_dispatch** with operators that take in an OptionalTensorList argument (#88887)
- Allow direct Tensor constructor to return preexisting PyObject (#92754)
- Add fallthrough kernel for AutogradMeta key (#94603)
- Several fixes to existing primtorch and reference decompositions:
  - `cat`: fix striding (#89332)
  - `prelu`: Fix prelu ref when a.ndim &lt; 2 (#89809)
  - `huber_loss_backward` fix (#86955)
  - `uniform` fix (#90094)
  - `unfold_copy` fix (#86371)
- Fix aliasing for primtorch view meta kernels (#86285)
- Properly compute device for elementwise operations with CPU scalar tensor (#93073)
- Several fixes to existing operators’ meta tensor kernels:
  - aten.\_embedding_bag (#92549)
  - aten.fill\_ (#87493)
  - `aten.group_norm` type promotion fix (#86607)
  - aten.\_cudnn_rnn (#91333)
  - aten.bernoulli (#88676)
  - unsqueeze\_ (#88675)
- Several bug fixes as part of hardening functionalization, which is used in AOTAutograd:
  - fix detach() in functionalization (#87750)
  - fix `torch.as_strided_scatter_backward` memory initialization (#88342)
  - fix functionalization resize stride compute (#94018)
  - fix x.is_contiguous(channels_last) in functionalization (#94195)
  - fix set\_() with functionalization (#90722)
  - check for undefined tensors in advanced indexing during functionalization (#90791)
  - fix some composite compliance ops for functionalization (#86470)
  - Make `aten.copy` preserve strides (#89464)

## Sparse API

- Fixes to `torch.mm`: (#90763), (#90917), (#91094)
- Fix CSR to CSC conversion when given indices of int32 dtype (#91061)
- Fix `mul` when given CUDA CSR Tensor and scalar (#91239)
- Fix conversion from CSC, BSC to COO to only result in coalesced Tensors when appropriate (#91440)
- Fix numel after resizing a CSR/BSR/CSC/BSC tensor. (#91831)
- Fix `torch.triangular_solve` for CSR on CPU when `unitriangular=True`. (#93352)

## Distributions

- Fix philox randn to follow standard normal distribution (#91945)

## Cpu

- Fix access to uninitialized memory in VSX vector functions (#89833)
- Fix buffer overflow from AddressSanitizer checks due to inaccurate bfloat16 representation of large integer (#89210)
- Make torch.histc ignore NaNs on CPU (consistent with CUDA) (#85870)
- Fix vectorized trigonometric functions for VSX (#86453)
- Call `symint::sizes()` instead of `sizes()` on convolution error messages. (#89549)
- Make `torch.linspace` result on CPU consistent with numpy (#89048)
- Remove variable_excluded_from_dispatch() assertion from mkldnncommon (#92168)
- `exponential_` few fixes (1) lambda > 0 (2) mkl kernel to continuous (3) better error log on dtype (#92891)
- Vectorize more stable complex division (#93277)
- `cauchy_` few fixes (1) check gamma > 0 (2) better dtype error log (#93314)

## Intel

- Fix CPU autocast for torch.cat due to the new type ITensorListRef (#87756)
- Add parameters check for torch.\_mkldnn_transpose (#85318)
- Fix build with Intel compiler due to c10/util/TypeIndex.h (#89610)

## Package

- Treat builtins as default extern module (#88385)
- Support pickle version 4 by adding missing ops (#90223)
- Check spec for module source before falling back to file in package exporter (#90258)

## Quantization

- Fix the call to get_executorch_backend_config (#86338)
- Fix weight_dtype and bias_dtype backend_config checks (#86719)
- Respect non_leaf_module_list for activation modules (#88498)
- Fix incorrect integer cast on histogram observer bounds (#90355)
- Improve numerical stability of HistogramObserver (#86522)
- Quant_min typo bugfix in utils.py (#88024)
- Fix fuse_func method overwrite (#87791)
- Fix get_default_qat_qconfig for PT 1.13 (#88876)
- Check the value of numel to avoid segfault (#81547)
- Fix mkldnn quantization issue for weight reorder error (#86876)
- Fix Memory Leak in QNNPACK QSoftmax Op (#89544)
- Copy MHA's batch_first attribute in prepare() (#91680)
- Fix for swap_custom_module_to_observer doing duplicate swaps on the same node.target (#91905)

## Fx

- Correctly restore pybind11 error_already_set (#93238)
- Remove proxy tensor's check for data dependent output (#93265)
- Make ShapeEnv deepcopy-able (#93403)
- Fix SubgraphMatcher for case of no anchor found (#86421)
- Fix for partitioner with symbolic shapes (#86425)
- Fix getitem in partitioner and make metadata storage more consistent (#87012)
- Fix magic method try reverse protocol (#88030)
- Fix FakeTensorProp on Module with Parameters or Buffers (#88700)
- Fix PassManager to not use a class variable mutable list (#89108)
- Prevent tracing when we track_tensor_tree (#89139)
- Make all `make_fx` invocations isolated (opaque to higher `make_fx` invocations) by default (#93290)
- Fix matching args in PatternMatcher (#94375)
- Allow FakeTensorProp to run on graphs traced with some None inputs (#94569)
- Copy codegen in legalize_graph (#90023)
- Fix proxy unwrapping for cond() (#91907)

## ONNX

- Fix `triu`/`tril` operator export with diagonal input (#86843)
- Skip tensor printing during model tracing (#86223)
- Fix `aten::index_put(self, mask, v)` export when `rank(mask) &lt; rank(self)` (#92862)
- Fix 0d-tensor broadcast export (#87211)
- Fix device type detection based on strings (#86168)
- Fix `scatter_add` with different static shape of src and index (#89787)
- Fix `_pad_circular` export (#86984)
- Fix concat with empty tensors (#87620)
- Disable ONNX `ceil_mode` and `count_include_pad` to align torch `ceil_mode` results in corner case (#87892)
- Fix ignored small eps in layer normalization in fp16 (#89869)
- Fix `unconvertible_ops` as per #89261 (#89299)
- Fix `Gather` replacement in `RNN peephole` (#93120)
- Fix `cat` operator for tensors with unknown rank (#94870)
- Fix scalar type analysis for copied constant (#86716)
- Fix scalar type detection for optional tensors (#94427)
- Fix 'prim::PackPadded' shape inference (#91829)
- Add `onnx::Max` into standard Op for scalar type alignment (#88750)
- Add `setType` from user into `InferredType` and `Reliable` in `ConstantValueMap` (#88622)
- Integrate ONNX ATen Fallback export with the new operator registry (#87735)
- Fix ONNX ATen Fallback integration for `BUILD_CAFFE2=0` builds (#88504)
- Fix `torch.autograd.Function.symbolic` method support (#94746)
- Fix `FindCommonAncestor` in `function_extraction` (#86650)
- Update training state logic to support `ScriptedModule` (#86745)

## ROCm

- Fix hipify mapping for cuDeviceGet (#90726)

## Mps

- Fix issues with non-contiguous Tensor handling (#86956, #86958)
- Fix issues with ops implementation `torch.median` (#90326, #88807), `torch.{std,var}` `correction` argument (#91203), `torch.index_select` (#94117, #91064), `torch.cumsum` (#94119), `torch.where` (#86240), `torch.nn.Embedding` (#82809), `torch.nn.Softplus` (#88555), `torch.nn.functional.pad` (#89864), `torch.max` (#91520), padding functions (#91522), `torch.nn.functional.upsample` (#91669), pooling functions (#91519, #94348), `torch.nn.{NLLLoss,SmoothL1Loss}` (#94226), `torch.nn.SoftPlus` (#94256), `torch.masked_fill` (#94263), `torch.fill_` (#94479), `torch.median` (#94489), `torch.nonzero` (#94442), `torch.nn.BatchNorm` (#94351), `torch.{min,max}` (#94386), `torch.nn.GELU` (#94529), `torch.nn.LSTM` (#94889), #95137),`torch.nn.Conv2d`(#95078),`torch.nn.functional.bilinear`(#94892),`torch.copy\_` (#95272),`torch.max_pool2d`(#94963),`torch.div` (#95769)
- Fix issues with `torch.bool` for Unary ops (#91120), scatter ops (#94464),
- Fix issues with `torch.float16` for `torch.nan_to_num` (#94220), `torch.nn.HuberLoss` (#94567)
- Properly raise error for `torch.int64` inputs for `torch.dot` (#94270), `torch.floor_divide` (#94488), `torch.square` (#94766),
- Properly cast `torch.int64` to `torch.int32` for reduction ops and raise warning. (#94484)
- Properly raise unimplemented error for `torch.nn.Conv3d` (#94492),
- Fix data type issues with index_add for non-`torch.float` inputs by casting them to `torch.float` (#88542)
- Fix the high watermark value for unified memory allocation on x86 (#91268)
- Fix handling of ops taking multiple dtypes as input (#91197, #91514)
- Fix handling of channels last for `torch.cat` (#91786, #94662), `torch.Conv2d` (#91822, #94384), `torch.nn.{ELU,ReLU,Hardswish}` (#94664), `torch.nn.BatchNorm` (#94760), `torch.nn.MaxPool2d` (#94877)
- Fix view operations handling (#94259, #94278,#95145, #95762, #95905)
- Fix numerical stability issues with various ops (#94889)
- Fix TORCH_WARN_ONCE (#95559) (#95559)

## Build

- Move incorrectly placed closing curly brace of `extern "C"` block (#87853)
- Set INTERFACE_LINK_DIRECTORIES on caffe2::mkl (#89359)
- Also include MKL_THREAD_LIB in link libraries for caffe2::mkl (#89378)
- Fix MSVC compiler error in basic_ops.h (#93322)
- Fix a bug that redefines \_\_STDC_FORMAT_MACROS (#89310)
- Fix ReplaceWithMaybeCopy test in OSS (#88099)

## Jit

- Fix out-of-bounds error in torch.jit.script for functions with many decorators (#87804)
- Assorted fixes for NNC cpu fuser (#85056, #86788, #88798, #89978)
- Set the correct size of aten tensor in presence of MKL-DNN padding (#86767)
- Fix Scalar(bool) handling in toIValue (#87179)

## Vulkan

- Fix an issue with Vulkan not being able to be compiled on Windows (#92207)
- Fix a possible empty vector dereference in the Vulkan optimization pass (#92918)

## Cudnn

- Fix cudnn RNN reproducibility issue (#90522)
- Fix `benchmark_limit` ignoring failed kernels in FIND (#91032)

## Releng

- Set nvfuser default to disabled, keep CI (#86369)
- Add manual cuda deps search logic (#90411)
- Workaround for NumPy builds that ship with a broken Dlpack deleter (#89759)
- Workaround MSVC ICE due to constexpr char\* template argument (#86288)
- Add define to fix issue with compatibility with latest Windows SDK (#85408)
- Remove invalid git option when updating submodules (#91132)

# Performance

## Python API

- Improve torch.lerp performance on cpu (#84845)
- Improve torch.istft performance (#88060)
- Call view within einsum to remediate MPS regression (#87135)
- Remove unnecessary calls to python builtins(#94323)
- Improve type hints for Module forward hooks (#92061)

## Autograd API

- Use in-place input accumulation fast path for dense Tensors. (#90217)

## torch.nn API

- Improve `functional.interpolate()` speed for `torch.channels_last` (#86361, #86361, #90302)
- Improve performance for `functional.multi_head_attention_forward()` (#93234, #89847)
- Improve performance for `TransformerEncoderLayer()` and `MultiheadAttention()` (#87377, #88488, #88831, #88854, #88970, #91171)
- Improve `SyncBatchNorm()` performance by using the right gathering ops (#89521)
- Improve `ConvTransposed2D()` CPU performance for `torch.{float32, bfloat16}` (#92530)
- Improve `functional.local_response_norm()` performance for 3d inputs (#91052)

## torch.func

- Add vmap batching rule for: `bitwise operators` (#91971), `nansum` & `nanmean` (#91372), `all` & `any` (#91966), `torch.linalg.vander` (#91749), `slogdet` (#86815), `torch.index_fill` (#91364), `narrow_copy` (#88130), `view_copy` (#88150), `greater_equal.Scaler` (#91324)

## Cuda

- Layer norm backward speed gain with warp shuffles (#87445, #87814)
- Avoid unnecessary type casts (#86086)
- Use `atomicAdd` for `bfloat16` in Ampere and above (#84981)

## Cpp API

- Vectorize torch.exp2 on CPU and add complex support (#92115)
- Add various performance fixes to c++ STL usage (#94034)

## NestedTensor API

- Improve performance for NestedTensor `torch.bmm`(#86856), (#85894)
- Remove unecessary check in `select_nested` (#89150)

## Distributed

- Do not call `pad` in no-padding case(#88769)

## Complex API

- Improve complex `lerp` performance (#84844)

## Mobile

- Passing serialized XNNPACK model by reference (#89089)
- Fix to add multiple outputs for the CoreML delegate (#88345)

## Sparse API

- Improve performance of `mul` when given COO (#86269)
- Improve `to(dtype)` support for all sparse compressed formats (#89055)
- Improve conversion of BSR/BSC to COO using `to_sparse` (#91389)
- Improve `sparse_mask` (#91964)
- Improve `to_dense` backward by removing redundant call to `coalesce` (#92001)
- Improve validation of CSR/CSC/BSR/BSC tensors for low dimensional inputs (#94048)
- Improve torch.sparse.sampled_addmm performance on CPU for CSR inputs (#90978)

## Optimizer API

- Improve foreach implementations by pre-grouping tensors to maximize fast path for `{Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, NAdam, RAdam, RMSProp, RProp, SGD}`(#92048, #92362, #92363, #92349, #92364, #92365, #92369, #92372, #92338)

## Cpu

- Optimizations for flip (#89414, #91806,#88989, #90013)
- Add fmsub to vectorization primitives (#86568)
- Optimize GELU BFloat16 Impl in CPU path (#79378)
- Fix `biasadd` OMP perf issue for the packed MKL SGEMM (#92300)
- Optimize LogSoftmax by improving thread-allocation in `_vec_log_softmax_lastdim` (#85398)
- BF16 autocast conv transpose 1d/2d/3d for CPU (#92527)
- Add mkl implementation for exponential on CPU (#69967)

## Fx

- Use deque instead of list for BFS (#91139)
- Refactor the dfs cyclic search from recursive to iterative approach (#91042)

## Mps

- Increase performance of `torch.add{cmul,cdiv,mm}`(#94214, #94534)`torch.multinomial` (#86342), faster op launch time (#86437), `torch.linear` (#91114), view handling (#91743, #94218), `convolutions`(#94661), `scatter/gather` (#94663)

## Jit

- Add BFloat16 dtype support for oneDNN Graph JIT fuser (#85591)

## Cudnn

- Improve hot path heuristics performance in V8 (#90811)

# Documentation

## Python API

- Fix various spelling and grammatical errors (#87357, #87583, #88033, #91641, #91871, #86642, #86721, #90110, #87724, #88483, #92049, #92762, #88962)
- Fix the documentation of various functions (#88059, #94545, #86593, #93145, #90071, #87870, #91627, #89910, #79086)
- Fix dev-discuss link in the maintainer docs (#89493)
- Add General Project Policies (#87385)

## Autograd API

- Improve autograd documentation (#89401, #93065)

## torch.nn API

- Improve documentation for: `MaxPool2d` (#86559), `utils.clip_grad_norm_()` (#91312), `Module()` (#87142), `{Unfold,Fold}()` (#88819), `torch.nn.functional.gelu` (#89061), `functional.conv2d` `padding` (#85004), `functional.leaky_relu()` (#94090), `MaxUnpool{1,2,3}D` (#94629)

## NestedTensor API

- Update Persons of Interest (#90069)
- Fix path to nested_tensor in example (#86891)

## Mps

- Add 'mps' to the tensor attributes doc page (#86585)

## Distributed

- Activation checkpointing
  - Clean up comments in activation checkpoint (#86622)
- Distributed (c10d)
  - Improve documentation for various functions (#87018, #94543, #91116,#89905, #86438 )
- DistributedDataParallel
  - Improve Documentation (#86221, #91832)
- RPC
  - Fix non-existing parameters in docstrings in benchmarks (#91115)
- Tensor parallelism and DTensor:
  - Add more clarifications and fix errors in tensor parallelism docs (#94786)
  - Update 2D parallelism API naming and docs (#94771)
- FullyShardedDataParallel
  - Add docs to explain the running the forward pass of of submodules in FSDP (#86343)
  - Clarify warnings to mention collectives (#87478)
  - Remove HSDP Zero-2 from doc (#90503)
  - Improve the comments for FSDP (#92359)
- Distributed Checkpoint
  - Enable documentation for Distributed Checkpoint. (#92813)
- Torch Elastic
  - Fix a minor typo in documentation (#90667)
  - Fix `torch.distributed.run` init connect timeout by comparing `host` with the current IP list (#90221)

## torch.func

- Downgrade the warning about forward-mode AD coverage (#87383)
- Add version selector back to functorch docs (#86602)
- Add documentation for torch.func (#91319)
- Fix AOTAutograd tutorial (#87415)
- Add migration guide from functorch (#91811)
- Improve inplace/view note on copy slices (#89856)
- Add more details to the functorch install page (#86823)

## Linalg API

- Add a note on the stability of linalg functions. (#88313)
- Improve documentation for various linalg functions (#89013,#89383, #91129)

## Composability

- Fix ScalarTensor **repr** in Extending PyTorch example (#86330)
- Fix incorrect wrapping of function decorator (#94446)
- Add **all** to torch.{autograd, fx, cuda} submodules (#85343)

## Dataloader API

- Update dataloader docstring mentioning prefetch factor behavior (#89874)

## Sparse API

- Extend documentation for `to_sparse` (#89912)
- Small correction to `torch.sparse` overview documentation(#93258)

## Optimizer API

- Improve documentation for various optimizers (#91195, #91196, #91881, #89575, #86629, #92111)
- Add general documentation on our algorithm defaults (#95391)

## Serialization

- Fix various spelling and grammatical errors (#90662, #91253)

## Distributions

- Improve documentation for various distributions (#91091, #87577)
- Add original sources/references to Wishart.py in distributions (#86543)

## Quantizationan

- Improvments to various READEMEs (#89319, #86914,#86523, #89795, #90403)
- Add docstrings for operators defined in torch.ops.quantized_decomposed namespace (#89547)
- Add x86 backend as default backend of server inference (#86794)
- Fix non-existing parameters in docstrings in torch/ao (#90875)
- Move parts of BackendConfig tutorial (#91999)

## ONNX

- Fix non-existing parameters in docstrings in torch/onnx (#90593)
- Update diagnostics system (#94565)

## Releng

- Enabled xdoctest runner in CI (#83816)
