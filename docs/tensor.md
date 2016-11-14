# Tensors

A `Tensor` is a potentially multi-dimensional matrix.
The number of dimensions is unlimited.

The `Tensor` set of classes are probably the most important class in
`torch`. Almost every package depends on these classes. They are *__the__*
class for handling numeric data. As with pretty much anything in
[torch], tensors are serializable with `torch.save` and `torch.load`

There are 7 Tensor classes in torch:

- `torch.FloatTensor`   :   Signed 32-bit floating point tensor
- `torch.DoubleTensor`  :   Signed 64-bit floating point tensor
- `torch.ByteTensor`    :   Signed  8-bit integer tensor
- `torch.CharTensor`    : Unsigned  8-bit integer tensor
- `torch.ShortTensor`   :   Signed 16-bit integer tensor
- `torch.IntTensor`     :   Signed 32-bit integer tensor
- `torch.LongTensor`    :   Signed 64-bit integer tensor

The data in these tensors lives on the system memory connected to your CPU.

Most numeric operations are implemented _only_ for `FloatTensor` and `DoubleTensor`.
Other Tensor types are useful if you want to save memory space or specifically
do integer operations.

The number of dimensions of a `Tensor` can be queried by
`ndimension()` or `dim()`. Size of the `i-th` dimension is
returned by `size(i)`. A tuple containing the size of all the dimensions
can be returned by `size()`.

```python
import torch

# allocate a matrix of shape 3x4
a = torch.FloatTensor(3, 4)
print(a)

# convert this into a LongTensor
b = a.long()
print(b)

# print the size of the tensor
print(a.size())

# print the number of dimensions
print(a.dim())
```

These tensors can be converted to numpy arrays very efficiently
with zero memory copies.
For this, the two provided functions are `.numpy()` and `torch.from_numpy()`

```python
import numpy as np

# convert to numpy
c = a.numpy()
print(type(c))
```

When using GPUs, each of the classes above has an equivalent
class such as: `torch.cuda.FloatTensor`, `torch.cuda.LongTensor`, etc.
When one allocates a CUDA tensor, the data in these tensors lives in the
GPU memory.

One can seamlessly transfer a tensor from the CPU to the GPU, as well as
between different GPUs on your machine.

Apart from the above 7 tensor types, there is one additional tensor type on the GPU

- `torch.cuda.HalfTensor` : Signed 16-bit floating point tensor

```python
import torch.cuda

# allocate a matrix of shape 3x4
a = torch.cuda.FloatTensor(3, 4)
print(a)

# transfer this to the CPU
b = a.cpu()
print(b)

# transfer this back to the GPU-1
a = b.cuda()
print(a)

# transfer this to GPU-2
b = a.cuda(1)
```

## Internal data representation

The actual data of a `Tensor` is contained into a
`Storage`. It can be accessed using
`storage()`. While the memory of a
`Tensor` has to be contained in this unique `Storage`, it might
not be contiguous: the first position used in the `Storage` is given
by `storage_offset()` (starting at `0`).
And the _jump_ needed to go from one element to another
element in the `i-th` dimension is given by
`stride(i-1)`. See the code example for an illustration.

```python
# given a 3d tensor
x = torch.FloatTensor(7,7,7)

# accessing the element `(3,4,5)` can be done by
x[3 - 1][4 - 1][5 - 1]
# or equivalently (but slowly!)
x.storage()[x.storageOffset()
            + (3 - 1) * x.stride(0)
			+ (4 - 1) * x.stride(1)
			+ (5 - 1) * x.stride(2)]
```

One could say that a `Tensor` is a particular way of _viewing_ a
`Storage`: a `Storage` only represents a chunk of memory, while the
`Tensor` interprets this chunk of memory as having dimensions:

```python
# a tensor interprets a chunk of memory as having dimensions
>>> x = torch.Tensor(4,5)
>>> s = x.storage()
>>> for i in range(s.size()): # fill up the Storage
>>>   s[i] = i

# s is interpreted by x as a 2D matrix
>>> print(x)

  1   2   3   4   5
  6   7   8   9  10
 11  12  13  14  15
 16  17  18  19  20
[torch.FloatTensor of dimension 4x5]
```

Note also that in Torch7 ___elements in the same row___ [elements along the __last__ dimension]
are contiguous in memory for a matrix [tensor]:

This is exactly like in `C` and `numpy` (and not `Fortran`).

## Default Tensor type

For convenience, _an alias_ `torch.Tensor` is provided, which allows the user to write
type-independent scripts, which can then ran after choosing the desired Tensor type with
a call like

`torch.set_default_tensor_type('torch.DoubleTensor')`


By default, the alias points to `torch.FloatTensor`.

## Efficient memory management

_All_ tensor operations post-fixed with an underscore (for example `.fill_`)
do _not_ make any memory copy. All these methods transform the existing tensor.
Tensor methods such as `narrow` and `select` return a new tensor referencing _the same storage_.
This magical behavior is internally obtained by good usage of the `stride()` and
`storage_offset()`. See the code example illustrating this.

```python
>>> x = torch.Tensor(5).zero_()
>>> print(x)
0
0
0
0
0
[torch.FloatTensor of dimension 5]
>>> x.narrow(0, 1, 2).fill_(1)
>>> # narrow() returns a Tensor referencing the same Storage as x
>>> print(x)
 0
 1
 1
 1
 0
[torch.FloatTensor of dimension 5]
```

If you really need to copy a `Tensor`, you can use the `copy_()` method:

```python
# making a copy of a tensor
y = x.new(x.size()).copy_(x)
y = x.clone()
```
Or the convenience method `clone()`

We now describe all the methods for `Tensor`. If you want to specify the Tensor type,
just replace `Tensor` by the name of the Tensor variant (like `CharTensor`).

## Constructors ##

Tensor constructors, create new Tensor object, optionally, allocating
new memory. By default the elements of a newly allocated memory are
not initialized, therefore, might contain arbitrary numbers. Here are
several ways to construct a new `Tensor`.

### torch.Tensor() ###

Returns an empty tensor.

### torch.Tensor(tensor) ###

Returns a new tensor which reference the same `Storage` than the given `tensor`.
The `size`, `stride`, and `storage_offset` are the same than the given tensor.

The new `Tensor` is now going to "view" the same `storage`
as the given `tensor`. As a result, any modification in the elements
of the `Tensor` will have a impact on the elements of the given
`tensor`, and vice-versa. No memory copy!

```python
>>> x = torch.Tensor(2,5).fill_(3.14)
>>> x
 3.1400  3.1400  3.1400  3.1400  3.1400
 3.1400  3.1400  3.1400  3.1400  3.1400
[torch.FloatTensor of dimension 2x5]

>>> y = torch.Tensor(x)
>>> y
 3.1400  3.1400  3.1400  3.1400  3.1400
 3.1400  3.1400  3.1400  3.1400  3.1400
[torch.FloatTensor of dimension 2x5]

>>> y.zero_()
>>> x # elements of x are the same as y!
0 0 0 0 0
0 0 0 0 0
[torch.FloatTensor of dimension 2x5]
```

### torch.Tensor(sz1 [,sz2 [,sz3 [,sz4 [,sz5 ...]]]]]) ###

Create a tensor of the given sizes.
The tensor size will be `sz1 x sz2 x sx3 x sz4 x sz5 x ...`.

### torch.Tensor(sizes) ###

Create a tensor of any number of dimensions. `sizes` gives the size in each dimension of
the tensor and is of type `torch.Size`. 

```python
Example, create a 4D 4x4x3x2 tensor:
x = torch.Tensor(torch.Size([4,4,3,2]))
```

### torch.Tensor(storage) ###

Returns a tensor which uses the existing `Storage` starting at a storage offset of 0.

### torch.Tensor(sequence) ###

One can create a tensor from a python sequence.

For example, you can create a `Tensor` from a `list` or a `tuple`

```python
# create a 2d tensor from a list of lists
>>> torch.Tensor([[1,2,3,4], [5,6,7,8]])
 1  2  3  4
 5  6  7  8
[torch.FloatTensor of dimension 2x4]
```

### torch.Tensor(ndarray) ###

Creates a `Tensor` from a NumPy `ndarray`.
If the `dtype` of the `ndarray` is the same as the type of the `Tensor` being created,
The underlying memory of both are shared, i.e. if the value of an element
in the `ndarray` is changed, the corresponding value in the `Tensor` changes,
and vice versa.

```python
# create a ndarray of dtype=int64
>>> a = np.random.randint(2, size=10)
>>> a
array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])
# create a LongTensor. Since they are the same type (int64), the memory is shared
>>> b = torch.LongTensor(a)
 0
 0
 1
 1
 0
 1
 1
 0
 0
 0
[torch.LongTensor of size 10]
>>> b[3] = 100
>>> print(a[3])
100

# now create an IntTensor from the same ndarray.
# The memory is not shared in this case as the dtype=int64 != IntTensor (int32)
>>> b = torch.IntTensor(a)
>>> b[3] = 30000
>>> print(a[3])
100
# a did not change to the value 30000
```

## NumPy Conversion ##
### torch.from_numpy(ndarray)

This is a convenience function similar to the constructor above.
Given a numpy `ndarray`, it constructs a torch `Tensor` of the same `dtype`
as the numpy array.

For example, passing in an ndarray of dtype=float64 will create a torch.DoubleTensor

### Tensor.numpy()

This is a member function on a tensor that converts a torch `Tensor` to a
numpy `ndarray`. The memory of the data of both objects is shared.
Hence, changing a value in the `Tensor` will change the corresponding value in
the `ndarray` and vice versa.

```python
>>> a = torch.randn(3,4)
>>> b = a.numpy() # creates a numpy array with dtype=float32 in this case
>>> print(a)
-1.0453  1.4730 -1.8990 -0.7763
 1.8155  1.4004 -1.5286  1.0420
 0.6551  1.0258  0.1152 -0.3239
[torch.FloatTensor of size 3x4]
>>> print(b)
[[-1.04525673  1.4730444  -1.89899576 -0.77626842]
 [ 1.81549406  1.40035892 -1.5286355   1.04199517]
 [ 0.6551016   1.02575183  0.11520521 -0.32391372]]
>>> a[2][2] = 1000
>>> print(b)
[[ -1.04525673e+00   1.47304440e+00  -1.89899576e+00  -7.76268423e-01]
 [  1.81549406e+00   1.40035892e+00  -1.52863550e+00   1.04199517e+00]
 [  6.55101597e-01   1.02575183e+00   1.00000000e+03  -3.23913723e-01]]
# notice that b[2][2] has changed to the value 1000 too.
```

### torch.is_tensor(obj)

Returns True if the passed-in object is a `Tensor` (of any type). Returns `False` otherwise.

### torch.is_storage

Returns True if the passed-in object is a `Storage` (of any type). Returns `False` otherwise.

### torch.expand_as
### torch.expand
### torch.view
### torch.view_as
### torch.permute
### torch.pin_memory
### copy
### split
### chunk
### tolist
### repeat
### unsqueeze
### unsqueeze_
### add, iadd, sub, isub, mul, imul, matmul, div, rdiv, idiv, mod, neg

## GPU Semantics ##

When you create a `torch.cuda.*Tensor`, it is allocated on the current GPU.
However, you could allocate it on another GPU as well, using the `with torch.cuda.device(id)` context.
All allocations within this context will be placed on the GPU `id`. 

Once `Tensor`s are allocated, you can do operations on them from any GPU context, and the results will be placed on the same device as where the source `Tensor` is located.

For example if Tensor `a` and `b` are on GPU-2, but the GPU-1 is the current device.
If one does `c = a + b`, then `c` will be on GPU-2, regardless of what the current device is.

Cross-GPU operations are not allowed. The only Cross-GPU operation allowed is `copy`.

If `a` is on GPU-1 and `b` is on GPU-2, then `c = a + b` will result in an error.

See the example for more clarity on these semantics.

```python
# Tensors are allocated on GPU 1 by default
x = torch.cuda.FloatTensor(1)
# x.get_device() == 0
y = torch.FloatTensor(1).cuda()
# y.get_device() == 0

with torch.cuda.device(1):
    # allocates a tensor on GPU 2
    a = torch.cuda.FloatTensor(1)

    # transfers a tensor from CPU to GPU-2
	b = torch.FloatTensor(1).cuda()
	# a.get_device() == b.get_device() == 1

    z = x + y
	# z.get_device() == 1

    # even within a context, you can give a GPU id to the .cuda call
    c = torch.randn(2).cuda(2)
	# c.get_device() == 2
	
```

