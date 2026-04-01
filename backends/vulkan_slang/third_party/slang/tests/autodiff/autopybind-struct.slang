//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none
//TEST:SIMPLE(filecheck=TORCH): -target torch -line-directive-mode none

[PyExport("Foo")]
struct Foo
{
    TensorView<float> inValues1;
    TensorView<float> inValues2;
}

// CUDA: __global__ void __kernel__myKernel(Foo_[[#]] {{[[:alnum:]_]+}}, TensorView {{[[:alnum:]_]+}})
[AutoPyBindCUDA]
[CudaKernel]
void myKernel(Foo foo, TensorView<float> outValues)
{
    if (cudaThreadIdx().x > 0)
        return;
    outValues.store(
        cudaThreadIdx().x,
        sin(foo.inValues1.load(cudaThreadIdx().x)) * cos(foo.inValues2.load(cudaThreadIdx().x)));
}

// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: __kernel__myKernel(Foo_[[#]] {{[[:alnum:]_]+}}, TensorView {{[[:alnum:]_]+}});
// 
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void myKernel(std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<torch::Tensor, torch::Tensor> {{[[:alnum:]_]+}}, torch::Tensor {{[[:alnum:]_]+}})
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: std::tuple<std::tuple<const char*, const char*, const char*, const char*>, std::tuple<const char*, const char*>, const char*, const char*> __funcinfo__myKernel()
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: std::tuple<std::tuple<const char*, const char*>, std::tuple<const char*, const char*>> __typeinfo__Foo()

