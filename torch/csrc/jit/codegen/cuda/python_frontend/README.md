# nvFuser Python Frontend

This frontend allows for a user to describe the set of operations for nvFuser to fuse via 1 or more kernels.  This frontend is intended to be an integration point with PyTorch or standalone applications.

# Usage

## Example 1 - Define and Execute a Fusion

```python
import torch
from torch._C._nvfuser import Fusion, FusionDefinition, DataType

fs = Fusion()
with FusionDefinition(fs) as fd :
    t0 = fd.define_tensor(symbolic_sizes=[-1, 1, -1],
                          contiguous=[True, True, True],
                          dtype=DataType.Float)
    t1 = fd.define_tensor(3)
    c0 = fd.define_constant(3.0)

    t2 = fd.ops.add(t0, t1)
    t3 = fd.ops.mul(t2, c0)
    t4 = fd.ops.sum(t3, [-1], False, DataType.Float)

    fd.add_output(t4)

input1 = torch.ones(2, 1, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')

nvf_out = fs.execute([input1, input2])[0]
```

## Example 2 - Lookup and Execute a `Fusion` Based on Id

```python
fid = 0
fs = Fusion(fid)

input1 = torch.ones(2, 1, 8, device='cuda')
input2 = torch.ones(2, 4, 8, device='cuda')

nvf_out = fs.execute([input1, input2])[0]
```

## Components

### `Fusion` - Represents a Fusion
#### `Fusion` Methods
* `defined()`: Allows you to query if the `Fusion` is already defined and can be executed.
* `execute([inputs])`:  Allows you to execute the currently defined fusion with a list of given inputs and returns a list of tensors.
* `id()`: Returns the fusion id for a given `Fusion`.
* `print()`: Prints the low level IR for the currently defined fusion.

### `FusionDefinition` Context Manager - Interface for Defining Fusions

#### Defining Input Tensors
_All intermediate tensors are created by operations.  Constant tensors do not exist._

There are 3 ways to define tensors that will be enumerated below.

##### 1.) Defining tensors by the number of input dimensions only
This interface tells nvFuser that the tensor has a given number of symbolic dimensions that are not necessarily contiguous in memory.  The user also has the ability to specify a data type.  The default type is `Float`.
```python
t0 = fd.define_tensor(3)
t1 = fd.define_tensor(3, DataType.Half)
```

##### 2.) Defining tensors by a list of concrete sizes and a list of strides
The `sizes` parameter defines the number of dimensions and the size of each dimension.  The `strides` parameter has to have the same number of dimensions as the `sizes` parameter.
nvFuser translates the concrete sizes and strides into symbolic sizes and contiguity information that can be directly defined via the next way to define tensors.  This allows the user to directly take a Pytorch defined tensor and query its sizes and strides in order to apply them in the definition.
```python
t0 = fd.define_tensor(sizes=[2, 4, 6], strides=[24, 6, 1], dtype=DataType.Half)
```

##### 3.) Defining tensors by a list of symbolic sizes and a list of contiguity information
The list of symbolic sizes defines the number of dimensions and `-1` is given for each dimension unless it is a broadcast dimension that is defined with a `1`.  The contiguity information is viewed from right to left.  A `True` definition indicates the current dimension is contiguous with the dimension to its right.

```python
t0 = fd.define_tensor(symbolic_sizes=[-1, 1, -1], contiguous=[True, True, True], dtype=DataType.Float)
```

#### Defining Input Scalars
_All intermediate scalars, except for constants, are created by operations._

The only thing the user has to define for a scalar is its type.

```python
s0 = fd.define_scalar(dtype=DataType.Half)
```

#### Defining Constant Scalars

Constants can be of types: `Bool`, `ComplexDouble`, `Double`, or `Int`.  The definition only takes a constant and the type is inferred by the constant.

```python
c0 = fd.define_constant(3.0)
```

#### Defining Operations

Operators are added with the following notation:
```python
output = fd.ops.foo(arg1, ... )
```
You can see a supported list of operations with the following query:
```python
python -c "from torch._C._nvfuser import FusionDefinition; help(FusionDefinition.Operators)"
```
#### Notating Outputs

The `FusionDefinition` `add_output` method is used to indicate an intermediate is an output to the fusion.

```python
add_output(output: Tensor)
# or
add_output(output: Scalar)
```

# Debug Information
**Query a list of supported operations:**
```python
python -c "from torch._C._nvfuser import FusionDefinition; help(FusionDefinition.Operators)"
```
**View the fusion definitions that are executed by setting an environment variable:**
```python
export PYTORCH_NVFUSER_DUMP=python_definition
```
Example Output:
```python
def nvfuser_fusion_id0(fd : FusionDefinition) -> None :
    T0 = fd.define_tensor(symbolic_sizes=[-1, 1, -1], contiguous=[True, True, True], dtype=DataType.Float)
    T1 = fd.define_tensor(symbolic_sizes=[-1, -1, -1], contiguous=[False, False, False], dtype=DataType.Float)
    S2 = fd.define_constant(3.00000)
    T3 = fd.ops.add(T0, T1)
    T4 = fd.ops.mul(T3, S2)
    T5 = fd.ops.sum(T4, axes=[-1], keepdim=False, dtype=DataType.Float)
    fd.add_output(T5)
```
