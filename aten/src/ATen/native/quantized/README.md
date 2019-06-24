The quantized folder holds the implementation of the low-level quantized ops and functions.
The ops are implemented using `c10`, and operate on the `at::QTensor` data type.

*Note that the name `QTensor` is only used to distinguish from a non-quantized `Tensor` type in the current discussion.
Otherwise, from the usage point of view, all quantized tensors are just `at::Tensor` types.*

This document serves as an entry point for quantized op implementation.

## Implementing native quantized ops

The new quantized ops are almost always located under the `ATen/native/quantized/cpu` folder.
For the sake of an example, let us implement an element-wise quantized logical AND operation under `ATen/native/quantized/cpu/qAND.cpp`.

### Step 0. Implement the quantized function

Before writing the quantized kernel and registering it, let us implement a quantized function.
That would assist in any further discussion.
The snippet below shows the implementation of a quantized AND operator, with the support of all implemented quantized types.

```c++
Tensor quantized_and(Tensor qa, Tensor qb) {
  // Some type checks for qa and qb should be here...
  Tensor qc;
  const auto scale = qa.q_scale().toDouble();
  const auto zero_point qa.q_zero_point().toLong()

  auto iter = TensorIterator::binary_op(qc, qa, qb);

  AT_DISPATCH_QINT_TYPES(qa.scalar_type(), "quantized_and", [&]() {
    Tensor qc = at::_empty_affine_quantized(qa.sizes(),
                                            at::device(kCPU).dtype(SCALAR_TYPE),
                                            scale, zero_point);
    binary_kernel(*iter, [&](scalar_t a_value, scalar_t b_value) -> scalar_t {
      return scalar_t(a_value.val_ & b_value.val_);
    });
  });
  return qc;
}
```

The code above is fairly straight-forward:
It takes two quantized tensors `qa` and `qb`, and uses `binary_kernel` to produce a quantized tensor `qc`.
The only part that (IMHO) requires explanation is the `AT_DISPATCH_QINT_TYPES`.
This macro makes sure that the underlying code works with all quantized types.
It provides several useful "aliases":

- `SCALAR_TYPE` -- quantized type option (i.e. `kQInt8`)
- `scalar_t` -- quantized data type (i.e. `qint8`)
- `underlying_t` -- underlying POD data type (i.e. `int8_t`)

The macro takes three arguments:
1. Quantized data type. This will define what the "aliases" are.
In the example above, the resulting tensor will be the same as the `qa.scalar_type()`.
2. Function name. This argument is currently used for error reporting.
3. Implementation lambda. The main implementation should sit in the body of this lambda.
it should also use the aliases for the quantized data types instead of the explicit data types.

### Step 1. Create the kernel

All kernels must be classes inheriting from `c10::OperatorKernel`.
The implementation itself should be under the `operator()` method.
In the `qAND.cpp` file, we create the following

```c++
class QuantizedAnd final : public c10::OperatorKernel {
 public:
  Tensor operator(Tensor qa, Tensor qb) {
    return quantized_and(qa, qb);
  }
};
```

### Step 2. Register the kernel

The registration is done using the `c10::RegisterOperators().op(...)`.

```c++
static auto registry = c10::RegisterOperators().op(
    "quantized::and(Tensor qa, Tensor qb) -> Tensor",
    c10::RegisterOperators::options()
      .kernel<QuantizedAnd>(QuantizedCPUTensorId()));
```

The registry takes two arguments:

1. **Function schema string**: This schema describes the usage of the op.
In the example above the schema is `"quantized::and(Tensor qa, Tensor qb) -> Tensor"`.
This translates to `torch._ops.ops.quantized.and` function in Python of the appropriate signature.
**Note:** The arguments signature in the schema is optional, and can also be written as `"quantized::and"` (without args).
2. **Registration options** should be of type `c10::RegisterOperators::options()`.
To attach a kernel to it, use `.kernel<KERNEL CLASS>(DISPATCHER)`.
In quantized ops you almost always want to use the `QuantizedCPUTensorId()` dispatcher.

### Putting it all together

The final file `ATen/native/quantized/cpu/qAND.cpp` would look as follows

```c++
#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

namespace at { namespace native {
namespace {

Tensor quantized_and(Tensor qa, Tensor qb) {
  // The awesome op implementation...
  return qc;
}

class QuantizedAnd final : public c10::OperatorKernel {
 public:
  Tensor operator(Tensor qa, Tensor qb) {
    return quantized_and(qa, qb);
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::and(Tensor qa, Tensor qb) -> Tensor",
    c10::RegisterOperators::options()
      .kernel<QuantizedAnd>(QuantizedCPUTensorId()));

}  // namespace
}}  // namespace at::native
```

Notice that we try to keep all the kernels in the anonymous namespace.
The reason for that is that we access the kernels only through the `torch` namespace.

### Step 3. Administrative stuff

Before the op can be used, it needs to be compiled.
If the op is placed under `native/quantized/cpu`, this already done for you.
However, if the location is changed, two files must be notified:

- *`caffe2/aten/TARGETS`* -- You can follow the same example, and add your path in somewhere in that file. Notice in this file we places the path to the quantized source files:
```bash
ATEN_NATIVE_CPP = glob([
  # ...
  "src/ATen/native/quantized/**/*.cpp",
])
```

- *`caffe2/aten/src/ATen/CMakeLists.txt`* -- Again, following the example, you must add your paths.
The current quantization paths are added as
```bash
FILE(GLOB native_quantized_cpp
          "native/quantized/*.cpp"
          "native/quantized/cpu/*.cpp")
```

## Using quantized ops

### Python

Usage in Python is pretty easy.
To implement the python quantized function using our kernel, you can do the following

```python
from torch._ops import ops

def quantized_and(qa, qb):
  # Notice the schema changed from `quantized::and` to `quantized.and`
  return ops.quantized.and(qa, qb)
```

### C++

You should not need to use the registered kernels in C++.
However, if you are feeling really brave, you can use the following

```c++
namespace at { namespace native {
namespace dispatch_tools {
/* Creates a stack of inputs consumable by the dispatcher.*/
template<class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

/* Given an operator handle, calls it using some arguments.*/
template<class... Args>
inline std::vector<c10::IValue> callOp(const c10::OperatorHandle& op,
                                       Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  auto kernel = c10::Dispatcher::singleton().lookup(op, &stack);
  kernel.call(&stack);
  return stack;
}

/* Finds the op and calls the callOp on it.*/
template<class... Args>
inline std::vector<c10::IValue> callOp(const char* func_name,
                                       const char* overload_name,
                                       Args... args) {
  const c10::optional<c10::OperatorHandle> op_handle
    = c10::Dispatcher::singleton().findSchema(func_name, overload_name);
  assert(op_handle.has_value());
  return callOp(op_handle.value(), args...);
}
}  // dispatch_tools

// This is your new function
Tensor quantized_and(Tensor qa, Tensor qb) {
  return dispatch_tools::callOp("quantized::and", "", qa, qb);
}
}}  // namespace at::native
```

The `dispatch_tools` is just a local namespace created for a sake of example.
