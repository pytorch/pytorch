//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none
//TEST:SIMPLE(filecheck=TORCH): -target torch -line-directive-mode none

// CUDA: __global__ void __kernel__myKernel_bwd_diff(DiffTensorView_[[#]] {{[[:alnum:]_]+}}, DiffTensorView_[[#]] {{[[:alnum:]_]+}})
// CUDA: __global__ void __kernel__myKernel_fwd_diff(DiffTensorView_[[#]] {{[[:alnum:]_]+}}, DiffTensorView_[[#]] {{[[:alnum:]_]+}})
// CUDA: __global__ void __kernel__myKernel(DiffTensorView_[[#]] {{[[:alnum:]_]+}}, DiffTensorView_[[#]] {{[[:alnum:]_]+}})

[AutoPyBindCUDA]
[Differentiable]
[CudaKernel]
void myKernel(DiffTensorView inValues, DiffTensorView outValues)
{
    if (cudaThreadIdx().x > 0)
        return;
    outValues[cudaThreadIdx().x] = sin(inValues[cudaThreadIdx().x]);
}

// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void __kernel__myKernel_bwd_diff(DiffTensorView_[[#]] {{[[:alnum:]_]+}}, DiffTensorView_[[#]] {{[[:alnum:]_]+}})
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void __kernel__myKernel_fwd_diff(DiffTensorView_[[#]] {{[[:alnum:]_]+}}, DiffTensorView_[[#]] {{[[:alnum:]_]+}})
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void __kernel__myKernel(DiffTensorView_[[#]] {{[[:alnum:]_]+}}, DiffTensorView_[[#]] {{[[:alnum:]_]+}})
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void myKernel(std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<torch::Tensor, std::tuple<torch::Tensor>> {{[[:alnum:]_]+}}, std::tuple<torch::Tensor, std::tuple<torch::Tensor>> {{[[:alnum:]_]+}})
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: std::tuple<std::tuple<const char*, const char*, const char*, const char*>, std::tuple<const char*, const char*>, const char*, const char*> __funcinfo__myKernel()
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void myKernel_fwd_diff(std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<torch::Tensor, std::tuple<torch::Tensor>> {{[[:alnum:]_]+}}, std::tuple<torch::Tensor, std::tuple<torch::Tensor>> {{[[:alnum:]_]+}})
//
// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void myKernel_bwd_diff(std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<torch::Tensor, std::tuple<torch::Tensor>> {{[[:alnum:]_]+}}, std::tuple<torch::Tensor, std::tuple<torch::Tensor>> {{[[:alnum:]_]+}})
//
// TORCH:     {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: std::tuple<std::tuple<const char*, const char*>, std::tuple<const char*, const char*>> __typeinfo__DiffTensorView()
//
// TORCH:     {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: std::tuple<std::tuple<const char*>, std::tuple<const char*>> __typeinfo__AtomicAdd()
//