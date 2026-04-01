---
layout: deprecated
permalink: "docs/user-guide/a1-02-slangpy"
---

Using Slang to Write PyTorch Kernels
=========================================================

> #### Deprecated Feature
> Note: This documentation is about `slang-torch`, an old way to use Slang with Python and PyTorch.
> Developers who are building new projects should use <a href="https://slangpy.shader-slang.org">SlangPy</a> instead.

If you are a PyTorch user seeking to write complex, high-performance, and automatically differentiated kernel functions using a per-thread programming model, we invite you to try Slang. Slang is a cutting-edge shading language that provides a straightforward way to define kernel functions that run incredibly fast in graphics applications. With the latest addition of automatic differentiation and PyTorch interop features, Slang offers an efficient solution for developing auto-differentiated kernels that run at lightning speed with a strongly typed, per-thread programming model.

One of the primary advantages of a per-thread programming model in kernel programming is the elimination of concerns regarding maintaining masks for branches. When developing a kernel in Slang, you can use all control flow statements, composite data types (structs, arrays, etc.), and function calls without additional effort. Code created with these language constructs can be automatically differentiated by the compiler without any restrictions. Additionally, Slang is a strongly typed language, which ensures that you will never encounter type errors at runtime. Most code errors can be identified as you type thanks to the [compiler's coding assistance service](https://marketplace.visualstudio.com/items?itemName=shader-slang.slang-language-extension), further streamlining the development process.

In addition, using a per-thread programming model also results in more optimized memory usage. When writing a kernel in Slang, most intermediate results do not need to be written out to global memory and then read back, reducing global memory bandwidth consumption and the delay caused by these memory operations. As a result, a Slang kernel can typically run at higher efficiency compared to the traditional bulk-synchronous programming model.

## Getting Started with SlangTorch

In this tutorial, we will use a simple example to walk through the steps to use Slang in your PyTorch project.

### Installation
`slangtorch` is available via PyPI, so you can install it simply through
```sh
pip install slangtorch
```

Note that `slangtorch` requires `torch` with CUDA support. See the [pytorch](https://pytorch.org/) installation page to find the right version for your platform.

You can check that you have the right installation by running: 
```sh
python -c "import torch; print(f'cuda: {torch.cuda.is_available()}')"
```

### Writing Slang kernels for `slangtorch` >= **v1.1.5**

From **v2023.4.0**, Slang supports auto-binding features that make it easier than ever to invoke Slang kernels from python, and interoperate seamlessly with `pytorch` tensors.

Here's a barebones example of a simple squaring kernel written in Slang (`square.slang`):

```csharp
[AutoPyBindCUDA]
[CUDAKernel]
void square(TensorView<float> input, TensorView<float> output)
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= input.size(0))
        return;

    output[dispatchIdx.x] = input[dispatchIdx.x] * input[dispatchIdx.x];
}

```

This code follows the standard pattern of a typical CUDA kernel function. It takes as input
two tensors, `input` and `output`. 
It first obtains the global dispatch index of the current thread and performs range check to make sure we don't read or write out
of the bounds of input and output tensors, and then calls `square()` to compute the per-element result, and
store it at the corresponding location in `output` tensor.


`slangtorch` works by compiling kernels to CUDA and it identifies the functions to compile by checking for the `[CUDAKernel]` attribute.
The second attribute `[AutoPyBindCUDA]` allows us to call `square` directly from python without having to write any host code. If you would like to write the host code yourself for finer control, see the other version of this example [here](#manually-binding-kernels).

You can now simply invoke this kernel from python:

```python
import torch
import slangtorch

m = slangtorch.loadModule('square.slang')

A = torch.randn((1024,), dtype=torch.float).cuda()

output = torch.zeros_like(A).cuda()

# Number of threads launched = blockSize * gridSize
m.square(input=A, output=output).launchRaw(blockSize=(32, 1, 1), gridSize=(64, 1, 1))

print(output)
```

The python script `slangtorch.loadModule("square.slang")` returns a scope that contains a handle to the `square` kernel.

The kernel can be invoked by 
1. calling `square` and binding `torch` tensors as arguments for the kernel, and then
2. launching it using `launchRaw()` by specifying CUDA launch arguments to `blockSize` & `gridSize`. (Refer to the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications) for restrictions around `blockSize`)

Note that for semantic clarity reasons, calling a kernel requires the use of keyword arguments with names that are lifted from the `.slang` implementation.

### Invoking derivatives of kernels using slangtorch

The `[AutoPyBindCUDA]` attribute can also be used on differentiable functions defined in Slang, and will automatically bind the derivatives. To do this, simply add the `[Differentiable]` attribute.

One key point is that the basic `TensorView<T>` objects are not differentiable. They can be used as buffers for data that does not require derivatives, or even as buffers for the manual accumulation of derivatives.

Instead, use the `DiffTensorView` type for when you need differentiable tensors. Currently, `DiffTensorView` only supports the `float` dtype variety.

Here's a barebones example of a differentiable version of `square`:

```csharp
[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void square(DiffTensorView input, DiffTensorView output)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    if (dispatchIdx.x >= input.size(0))
        return;
    
    output[dispatchIdx.x] = input[dispatchIdx.x] * input[dispatchIdx.x];
}
```

Now, `slangtorch.loadModule("square.slang")` returns a scope with three callable handles `square`, `square.fwd` for the forward-mode derivative & `square.bwd` for the reverse-mode derivative.

You can invoke `square()` normally to get the same effect as the previous example, or invoke `square.fwd()` / `square.bwd()` by binding pairs of tensors to compute the derivatives.


```python
import torch
import slangtorch

m = slangtorch.loadModule('square.slang')

input = torch.tensor((0, 1, 2, 3, 4, 5), dtype=torch.float).cuda()
output = torch.zeros_like(input).cuda()

# Invoke normally
m.square(input=input, output=output).launchRaw(blockSize=(6, 1, 1), gridSize=(1, 1, 1))

print(output)

# Invoke reverse-mode autodiff by first allocating tensors to hold the gradients
input = torch.tensor((0, 1, 2, 3, 4, 5), dtype=torch.float).cuda()
input_grad = torch.zeros_like(input).cuda()

output = torch.zeros_like(input)
# Pass in all 1s as the output derivative for our example
output_grad = torch.ones_like(output) 

m.square.bwd(
    input=(input, input_grad), output=(output, output_grad)
).launchRaw(
    blockSize=(6, 1, 1), gridSize=(1, 1, 1))

# Derivatives get propagated to input_grad
print(input_grad)

# Note that the derivatives in output_grad are 'consumed'.
# i.e. all zeros after the call.
print(output_grad)
```

`slangtorch` also binds the forward-mode version of your kernel (propagate derivatives of inputs to the output) which can be invoked the same way using `module.square.fwd()`

You can refer to [this documentation](autodiff) for a detailed reference of Slang's automatic differentiation feature.

### Wrapping your kernels as pytorch functions

`pytorch` offers an easy way to define a custom operation using `torch.autograd.Function`, and defining the `.forward()` and `.backward()` members.

This can be a very helpful way to wrap your Slang kernels as pytorch-compatible operations. Here's an example of the `square` kernel as a differentiable pytorch function.

```python
import torch
import slangtorch

m = slangtorch.loadModule("square.slang")

class MySquareFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.zeros_like(input)

        kernel_with_args = m.square(input=input, output=output)
        kernel_with_args.launchRaw(
            blockSize=(32, 32, 1),
            gridSize=((input.shape[0] + 31) // 32, (input.shape[1] + 31) // 32, 1))

        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input, output) = ctx.saved_tensors

        input_grad = torch.zeros_like(input)
        
        # Note: When using DiffTensorView, grad_output gets 'consumed' during the reverse-mode.
        # If grad_output may be reused, consider calling grad_output = grad_output.clone()
        #
        kernel_with_args = m.square.bwd(input=(input, input_grad), output=(output, grad_output))
        kernel_with_args.launchRaw(
            blockSize=(32, 32, 1),
            gridSize=((input.shape[0] + 31) // 32, (input.shape[1] + 31) // 32, 1))
        
        return input_grad
```

Now we can use the autograd function `MySquareFunc` in our python script:

```python
x = torch.tensor((3.0, 4.0), requires_grad=True, device='cuda')
print(f"X = {x}")
y_pred = MySquareFunc.apply(x)
loss = y_pred.sum()
loss.backward()
print(f"dX = {x.grad.cpu()}")
```

Output:
```
X = tensor([3., 4.],
           device='cuda:0', requires_grad=True)
dX = tensor([6., 8.])
```

And that's it! `slangtorch.loadModule` uses JIT compilation to compile your Slang source into CUDA binary.
It may take a little longer the first time you execute the script, but the compiled binaries will be cached and as long as the kernel code is not changed, future runs will not rebuild the CUDA kernel.

Because the PyTorch JIT system requires `ninja`, you need to make sure `ninja` is installed on your system
and is discoverable from the current environment, you also need to have a C++ compiler available on the system.
On Windows, this means that Visual Studio need to be installed.

## Specializing shaders using slangtorch

`slangtorch.loadModule` allows specialization parameters to be specified since it might be easier to write shaders with placeholder definitions that can be substituted at load-time.
For instance, here's a sphere tracer that uses a _compile-time_ specialization parameter for its maximum number of steps (`N`):

```csharp
float sphereTrace<let N:int>(Ray ray, SDF sdf)
{
    var pt = ray.o;
    for (int i = 0; i < N; i++)
    {
        pt += sdf.eval(pt) * ray.d;
    }

    return pt;
}

float render(Ray ray)
{
    // Use N=20 for sphere tracing.
    float3 pt = sphereTrace<20>(ray, sdf);
    return shade(pt, sdf.normal());
}
```

However, instead of using a fixed `20` steps, the renderer can be configured to use an arbitrary compile-time constant.

```csharp
// Compile-time constant. Expect "MAX_STEPS" to be set by the loadModule call.
static const uint kMaxSteps = MAX_STEPS;

float render(Ray ray)
{
    float3 pt = sphereTrace<kMaxSteps>(ray, sdf);
    return shade(pt, sdf.normal());
}
```

Then multiple versions of this shader can be compiled from Python using the `defines` argument:
```python
import slangtorch

sdfRenderer20Steps = slangtorch.loadModule('sdf.slang', defines={"MAX_STEPS": 20})
sdfRenderer50Steps = slangtorch.loadModule('sdf.slang', defines={"MAX_STEPS": 50})
...
```

This is often helpful for code re-use, parameter sweeping, comparison/ablation studies, and more, from the convenience of Python.

## Back-propagating Derivatives through Complex Access Patterns

In most common scenarios, a kernel function will access input tensors in a complex pattern instead of mapping
1:1 from an input element to an output element, like the `square` example shown above. When you have a kernel
function that access many different elements from the input tensors and use them to compute an output element,
the derivatives of each input element can't be represented directly as a function parameter, like the `x` in `square(x)`.

Consider a 3x3 box filtering kernel that computes for each pixel in a 2D image, the average value of its 
surrounding 3x3 pixel block. We can write a Slang function that computes the value of an output pixel:
```csharp
float computeOutputPixel(TensorView<float> input, uint2 pixelLoc)
{
    int width = input.size(0);
    int height = input.size(1);

    // Track the sum of neighboring pixels and the number
    // of pixels currently accumulated.
    int count = 0;
    float sumValue = 0.0;

    // Iterate through the surrounding area.
    for (int offsetX = -1; offsetX <= 1; offsetX++)
    {
        // Skip out of bounds pixels.
        int x = pixelLoc.x + offsetX;
        if (x < 0 || x >= width) continue;

        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            int y = pixelLoc.y + offsetY;
            if (y < 0 || y >= height) continue;
            sumValue += input[x, y];
            count++;
        }
    }

    // Compute the average value.
    sumValue /= count;

    return sumValue;
}
```

We can define our kernel function to compute the entire output image by calling `computeOutputPixel`:

```csharp
[CudaKernel]
void boxFilter_fwd(TensorView<float> input, TensorView<float> output)
{
    uint2 pixelLoc = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).xy;
    int width = input.dim(0);
    int height = input.dim(1);
    if (pixelLoc.x >= width) return;
    if (pixelLoc.y >= height) return;

    float outputValueAtPixel = computeOutputPixel(input, pixelLoc)

    // Write to output tensor.
    output[pixelLoc] = outputValueAtPixel;
}
```

How do we define the backward derivative propagation kernel? Note that in this example, there
isn't a function like `square` that we can just mark as `[Differentiable]` and
call `bwd_diff(square)` to get back the derivative of an input parameter.

In this example, the input comes from multiple elements in a tensor. How do we propagate the
derivatives to those input elements?

The solution is to wrap tensor access with a custom function:
```csharp
float getInputElement(
    TensorView<float> input,
    TensorView<float> inputGradToPropagateTo,
    uint2 loc)
{
    return input[loc];
}
```

Note that the `getInputElement` function simply returns `input[loc]` and is not using the
`inputGradToPropagateTo` parameter. That is intended. The `inputGradToPropagateTo` parameter
is used to hold the backward propagated derivatives of each input element, and is reserved for later use.

Now we can replace all direct accesses to `input` with a call to `getInputElement`. The
`computeOutputPixel` can be implemented as following:

```csharp
[Differentiable]
float computeOutputPixel(
    TensorView<float> input,
    TensorView<float> inputGradToPropagateTo,
    uint2 pixelLoc)
{
    int width = input.dim(0);
    int height = input.dim(1);

    // Track the sum of neighboring pixels and the number
    // of pixels currently accumulated.
    int count = 0;
    float sumValue = 0.0;

    // Iterate through the surrounding area.
    for (int offsetX = -1; offsetX <= 1; offsetX++)
    {
        // Skip out of bounds pixels.
        int x = pixelLoc.x + offsetX;
        if (x < 0 || x >= width) continue;

        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            int y = pixelLoc.y + offsetY;
            if (y < 0 || y >= height) continue;
            sumValue += getInputElement(input, inputGradToPropagateTo, uint2(x, y));
            count++;
        }
    }

    // Compute the average value.
    sumValue /= count;

    return sumValue;
}
```

The main changes compared to our original version of `computeOutputPixel` are:
- Added a `inputGradToPropagateTo` parameter.
- Modified `input[x,y]` with a call to `getInputElement`.
- Added a `[Differentiable]` attribute to the function.

With that, we can define our backward kernel function:

```csharp
[CudaKernel]
void boxFilter_bwd(
    TensorView<float> input,
    TensorView<float> resultGradToPropagateFrom,
    TensorView<float> inputGradToPropagateTo)
{
    uint2 pixelLoc = (cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx()).xy;
    int width = input.dim(0);
    int height = input.dim(1);
    if (pixelLoc.x >= width) return;
    if (pixelLoc.y >= height) return;

    bwd_diff(computeOutputPixel)(input, inputGradToPropagateTo, pixelLoc);
}
```

The kernel function simply calls `bwd_diff(computeOutputPixel)` without taking any return values from the call
and without writing to any elements in the final `inputGradToPropagateTo` tensor. But when exactly does the propagated
output get written to the output gradient tensor (`inputGradToPropagateTo`)?

And that logic is defined in our final piece of code:
```csharp
[BackwardDerivativeOf(getInputElement)]
void getInputElement_bwd(
    TensorView<float> input,
    TensorView<float> inputGradToPropagateTo,
    uint2 loc,
    float derivative)
{
    float oldVal;
    inputGradToPropagateTo.InterlockedAdd(loc, derivative, oldVal);
}
```

Here, we are providing a custom defined backward propagation function for `getInputElement`.
In this function, we simply add `derivative` to the element in `inputGradToPropagateTo` tensor.

When we call `bwd_diff(computeOutputPixel)` in `boxFilter_bwd`, the Slang compiler will automatically
differentiate all operations and function calls in `computeOutputPixel`. By wrapping the tensor element access
with `getInputElement` and by providing a custom backward propagation function of `getInputElement`, we are effectively
telling the compiler what to do when a derivative propagates to an input tensor element. Inside the body
of `getInputElement_bwd`, we define what to do then: atomically adds the derivative propagated to the input element
in the `inputGradToPropagateTo` tensor. Therefore, after running `boxFilter_bwd`, the `inputGradToPropagateTo` tensor will contain all the
back propagated derivative values.

Again, to understand all the details of the automatic differentiation system, please refer to the 
[Automatic Differentiation](autodiff) chapter for a detailed explanation.

## Manually binding kernels
`[AutoPyBindCUDA]` works for most use cases, but in certain situations, it may be necessary to write the *host* function by hand. The host function can also be written in Slang, and `slangtorch` handles its compilation to C++.

Here's the same `square` example from before:

```csharp
// square.slang
float compute_square(float x)
{
    return x * x;
}

[CudaKernel]
void square_kernel(TensorView<float> input, TensorView<float> output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= input.size(0))
        return;

    float result = compute_square(input[globalIdx.x]);

    output[globalIdx.x] = result;
}
```

To manually invoke this kernel, we then need to write a CPU(host) function that defines how this kernel is dispatched. This can be defined in the same Slang file:

```csharp
[TorchEntryPoint]
TorchTensor<float> square(TorchTensor<float> input)
{
    var result = TorchTensor<float>.zerosLike(input);
    let blockCount = uint3(1);
    let groupSize = uint3(result.size(0), result.size(1), 1);
    __dispatch_kernel(square_kernel, blockCount, groupSize)(input, result);
    return result;
}
```

Here, we mark the function with the `[TorchEntryPoint]` attribute, so it will be compiled to C++ and exported as a python callable. 
Since this is a host function, we can perform tensor allocations. For instance, `square()` calls `TorchTensor<float>.zerosLike` to allocate a 2D-tensor that has the same size as the input.
`zerosLike` returns a `TorchTensor<float>` object that represents a CPU handle of a PyTorch tensor.

Then we launch `square_kernel` with the `__dispatch_kernel` syntax. Note that we can directly pass
`TorchTensor<float>` arguments to a `TensorView<float>` parameter and the compiler will automatically convert the type and obtain a view into the tensor that can be accessed by the GPU kernel function.

### Calling a `[TorchEntryPoint]` function from Python

You can use the following code to call `square` from Python:

```python
import torch
import slangtorch

m = slangtorch.loadModule("square.slang")

x = torch.randn(2,2)
print(f"X = {x}")
y = m.square(x)
print(f"Y = {y.cpu()}")
```

Result output:
```
X = tensor([[ 0.1407,  0.6594],
        [-0.8978, -1.7230]])
Y = tensor([[0.0198, 0.4349],
        [0.8060, 2.9688]])
```

### Manual binding for kernel derivatives

The above example demonstrates how to write a simple kernel function in Slang and call it from Python.
Another major benefit of using Slang is that the Slang compiler support generating backward derivative
propagation functions automatically.

In the following section, we walk through how to use Slang to generate a backward propagation function
for `square`, and expose it to PyTorch as an autograd function.

First we need to tell Slang compiler that we need the `square` function to be considered a differentiable function, so Slang compiler can generate a backward derivative propagation function for it:
```csharp
[Differentiable]
float square(float x)
{
    return x * x;
}
```
This is done by simply adding a `[Differentiable]` attribute to our `square` function.

With that, we can now define `square_bwd_kernel` that performs backward propagation as:

```csharp
[CudaKernel]
void square_bwd_kernel(TensorView<float> input, TensorView<float> grad_out, TensorView<float> grad_propagated)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x >= input.size(0) || globalIdx.y >= input.size(1))
        return;

    DifferentialPair<float> dpInput = diffPair(input[globalIdx.xy]);
    var gradInElem = grad_out[globalIdx.xy];
    bwd_diff(square)(dpInput, gradInElem);
    grad_propagated[globalIdx.xy] = dpInput.d;
}
```

Note that the function follows the same structure of `square_fwd_kernel`, with the only difference being that
instead of calling into `square` to compute the forward value for each tensor element, we are calling `bwd_diff(square)`
that represents the automatically generated backward propagation function of `square`.
`bwd_diff(square)` will have the following signature:
```csharp
void bwd_diff_square(inout DifferentialPair<float> dpInput, float dOut);
```

Where the first parameter, `dpInput` represents a pair of original and derivative value for `input`, and the second parameter,
`dOut`, represents the initial derivative with regard to some latent variable that we wish to back-prop through. The resulting
derivative will be stored in `dpInput.d`. For example:

```csharp
// construct a pair where the primal value is 3, and derivative value is 0.
var dp = diffPair(3.0);
bwd_diff(square)(dp, 1.0);
// dp.d is now 6.0
```

Similar to `square_fwd`, we can define the host side function `square_bwd` as:

```csharp
[TorchEntryPoint]
TorchTensor<float> square_bwd(TorchTensor<float> input, TorchTensor<float> grad_out)
{
    var grad_propagated = TorchTensor<float>.zerosLike(input);
    let blockCount = uint3(1);
    let groupSize = uint3(input.size(0), input.size(1), 1);
    __dispatch_kernel(square_bwd_kernel, blockCount, groupSize)(input, grad_out, grad_propagated);
    return grad_propagated;
}
```

## Builtin Library Support for PyTorch Interop

As shown in previous tutorial, Slang has defined the `TorchTensor<T>` and `TensorView<T>` type for interop with PyTorch
tensors. The `TorchTensor<T>` represents the CPU view of a tensor and provides methods to allocate a new tensor object.
The `TensorView<T>` represents the GPU view of a tensor and provides accessors to read write tensor data.

Following is a list of built-in methods and attributes for PyTorch interop.

### `TorchTensor` methods

#### `static TorchTensor<T> TorchTensor<T>.alloc(uint x, uint y, ...)`
Allocates a new PyTorch tensor with the given dimensions. If `T` is a vector type, the length of the vector is implicitly included as the last dimension.
For example, `TorchTensor<float3>.alloc(4, 4)` allocates a 3D tensor of size `(4,4,3)`.

#### `static TorchTensor<T> TorchTensor<T>.emptyLike(TorchTensor<T> other)`
Allocates a new PyTorch tensor that has the same dimensions as `other` without initializing it.

#### `static TorchTensor<T> TorchTensor<T>.zerosLike(TorchTensor<T> other)`
Allocates a new PyTorch tensor that has the same dimensions as `other` and initialize it to zero.

#### `uint TorchTensor<T>.dims()`
Returns the tensor's dimension count.

#### `uint TorchTensor<T>.size(int dim)`
Returns the tensor's size (in number of elements) at `dim`.

#### `uint TorchTensor<T>.stride(int dim)`
Returns the tensor's stride (in bytes) at `dim`.

### `TensorView` methods

#### `TensorView<T>.operator[uint x, uint y, ...]`
Provide an accessor to data content in a tensor.

#### `TensorView<T>.operator[vector<uint, N> index]`
Provide an accessor to data content in a tensor, indexed by a uint vector.
`tensor[uint3(1,2,3)]` is equivalent to `tensor[1,2,3]`.

#### `uint TensorView<T>.dims()`
Returns the tensor's dimension count.

#### `uint TensorView<T>.size(int dim)`
Returns the tensor's size (in number of elements) at `dim`.

#### `uint TensorView<T>.stride(int dim)`
Returns the tensor's stride (in bytes) at `dim`.

#### `void TensorView<T>.fillZero()`
Fills the tensor with zeros. Modifies the tensor in-place.

#### `void TensorView<T>.fillValue(T value)`
Fills the tensor with the specified value, modifies the tensor in-place.

#### `T* TensorView<T>.data_ptr_at(vector<uint, N> index)`
Returns a pointer to the element at `index`.

#### `void TensorView<T>.InterlockedAdd(vector<uint, N> index, T val, out T oldVal)`
Atomically add `val` to element at `index`. 

#### `void TensorView<T>.InterlockedMin(vector<uint, N> index, T val, out T oldVal)`
Atomically computes the min of `val` and the element at `index`. Available for 32 and 64 bit integer types only.

#### `void TensorView<T>.InterlockedMax(vector<uint, N> index, T val, out T oldVal)`
Atomically computes the max of `val` and the element at `index`. Available for 32 and 64 bit integer types only.

#### `void TensorView<T>.InterlockedAnd(vector<uint, N> index, T val, out T oldVal)`
Atomically computes the bitwise and of `val` and the element at `index`. Available for 32 and 64 bit integer types only.

#### `void TensorView<T>.InterlockedOr(vector<uint, N> index, T val, out T oldVal)`
Atomically computes the bitwise or  of `val` and the element at `index`. Available for 32 and 64 bit integer types only.

#### `void TensorView<T>.InterlockedXor(vector<uint, N> index, T val, out T oldVal)`
Atomically computes the bitwise xor  of `val` and the element at `index`. Available for 32 and 64 bit integer types only.

#### `void TensorView<T>.InterlockedExchange(vector<uint, N> index, T val, out T oldVal)`
Atomically swaps `val` into the element at `index`. Available for `float` and 32/64 bit integer types only.

#### `void TensorView<T>.InterlockedCompareExchange(vector<uint, N> index, T compare, T val)`
Atomically swaps `val` into the element at `index` if the element equals to `compare`. Available for `float` and 32/64 bit integer types only.

### `DiffTensorView` methods

#### `DiffTensorView.operator[uint x, uint y, ...]`
Provide an accessor to data content in a tensor. This method is **differentiable**, and has the same semantics as using a `.load()` to get data, and `.store()` to set data.

#### `DiffTensorView.operator[vector<uint, N> index]`
Provide an accessor to data content in a tensor, indexed by a uint vector.`tensor[uint3(1,2,3)]` is equivalent to `tensor[1,2,3]`. This method is **differentiable**, and has the same semantics as using a `.load()` to get data, and `.store()` to set data.

#### `float DiffTensorView.load(vector<uint, N> index)`
Loads the 32-bit floating point data at the specified multi-dimensional `index`. This method is **differentiable**, and in reverse-mode will perform an atomic-add.

#### `void DiffTensorView.store(vector<uint, N> index, float val)`
Stores the 32-bit floating point value `val` at the specified multi-dimensional `index`. This method is **differentiable**, and in reverse-mode will perform an *atomic exchange* to retrieve the derivative and replace with 0.

#### `float DiffTensorView.loadOnce(vector<uint, N> index)`
Loads the 32-bit floating point data at the specified multi-dimensional `index`. This method is **differentiable**, and uses a simple `store` for the reverse-mode for faster gradient aggregation, but `loadOnce` **must** be used at most once per index. `loadOnce` is ideal for situations where each thread loads data from a unique index, but will cause incorrect gradients when an index may be accessed multiple times.

#### `void DiffTensorView.storeOnce(vector<uint, N> index, float val)`
Stores the 32-bit floating point value `val` at the specified multi-dimensional `index`. This method is **differentiable**, and uses a simple `load` for the reverse-mode for faster gradient loading, but `storeOnce` **must** be used at most once per index. `loadOnce` is ideal for situations where each thread stores data to a unique index, but will cause incorrect gradient propagation when an index may be accessed multiple times.

#### `uint DiffTensorView.size(int dim)`
Returns the underlying primal tensor's size (in number of elements) at `dim`.

#### `uint DiffTensorView.dims()`
Returns the underlying primal tensor's dimension count.

#### `uint DiffTensorView.stride(uint dim)`
Returns the stride of the underlying primal tensor's `dim` dimension

### CUDA Support Functions

#### `cudaThreadIdx()`
Returns the `threadIdx` variable in CUDA.

#### `cudaBlockIdx()`
Returns the `blockIdx` variable in CUDA.

#### `cudaBlockDim()`
Returns the `blockDim` variable in CUDA.

#### `syncTorchCudaStream()`
Waits for all pending CUDA kernel executions to complete on host.

### Attributes for PyTorch Interop

#### `[CudaKernel]` attribute
Marks a function as a CUDA kernel (maps to a `__global__` function)

#### `[TorchEntryPoint]` attribute
Marks a function for export to Python. Functions marked with `[TorchEntryPoint]` will be accessible from a loaded module returned by `slangtorch.loadModule`.

#### `[CudaDeviceExport]` attribute
Marks a function as a CUDA device function, and ensures the compiler to include it in the generated CUDA source.

#### `[AutoPyBindCUDA]` attribute
Marks a cuda kernel for automatic binding generation so that it may be invoked from python without having to hand-code the torch entry point. The marked function **must** also be marked with `[CudaKernel]`. If the marked function is also marked with `[Differentiable]`, this will also generate bindings for the derivative methods.

Restriction: methods marked with `[AutoPyBindCUDA]` will not operate 

## Type Marshalling Between Slang and Python


### Python-CUDA type marshalling for functions using `[AutoPyBindCUDA]` 

When using auto-binding, aggregate types like structs are converted to Python `namedtuples` and are made available when using `slangtorch.loadModule`. 

```csharp
// mesh.slang
struct Mesh
{
    TensorView<float> vertices;
    TensorView<int> indices;
};

[AutoPyBindCUDA]
[CUDAKernel]
void processMesh(Mesh mesh)
{
    /* ... */ 
}
```

Here, since `Mesh` is being used by `renderMesh`, the loaded module will provide `Mesh` as a python `namedtuple` with named fields.
While using the `namedtuple` is the best way to use structured arguments, they can also be passed as a python `dict` or `tuple`

```python
m = slangtorch.loadModule('mesh.slang')

vertices = torch.tensor()
indices = torch.tensor()

# use namedtuple to provide structured input.
mesh = m.Mesh(vertices=vertices, indices=indices)
m.processMesh(mesh=mesh).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))

# use dict to provide input.
mesh = {'vertices': vertices, 'indices':indices}
m.processMesh(mesh=mesh).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))

# use tuple to provide input (warning: user responsible for right order)
mesh = (vertices, indices)
m.processMesh(mesh=mesh).launchRaw(blockSize=(32, 32, 1), gridSize=(1, 1, 1))
```


### Python-CUDA type marshalling for functions using `[TorchEntryPoint]`

The return types and parameters types of an exported `[TorchEntryPoint]` function can be a basic type (e.g. `float`, `int` etc.), a vector type (e.g. `float3`), a `TorchTensor<T>` type, an array type, or a struct type.

When you use struct or array types in the function signature, it will be exposed as a Python tuple.
For example,
```csharp
struct MyReturnType
{
    TorchTensor<T> tensors[3];
    float v;
}

[TorchEntryPoint]
MyReturnType myFunc()
{
    ...
}
```

Calling `myFunc` from python will result in a python tuple in the form of
```
[[tensor, tensor, tensor], float]
```

The same transform rules apply to parameter types.
