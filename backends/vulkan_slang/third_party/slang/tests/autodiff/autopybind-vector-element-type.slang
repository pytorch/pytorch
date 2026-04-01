//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none
//TEST:SIMPLE(filecheck=TORCH): -target torch -line-directive-mode none

// CUDA: __global__ void __kernel__myKernel(TensorView inValues_[[#]], TensorView outValues_[[#]])
[AutoPyBindCUDA]
[CudaKernel]
void myKernel(TensorView<float> inValues, TensorView<float2> outValues)
{
    if (cudaThreadIdx().x > 0)
        return;
    outValues.store(cudaThreadIdx().x, sin(inValues.load(cudaThreadIdx().x)));
}

// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void __kernel__myKernel(TensorView {{[[:alnum:]_]+}}, TensorView {{[[:alnum:]_]+}});

// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void myKernel(std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, std::tuple<uint32_t, uint32_t, uint32_t> {{[[:alnum:]_]+}}, torch::Tensor {{[[:alnum:]_]+}}, torch::Tensor {{[[:alnum:]_]+}})

// TORCH:      TensorView {{[[:alnum:]_]+}} = make_tensor_view({{[[:alnum:]_]+}}, "outValues", torch::kFloat32, true);

// TORCH:      TensorView {{[[:alnum:]_]+}} = make_tensor_view({{[[:alnum:]_]+}}, "inValues", torch::kFloat32, false);

// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: std::tuple<std::tuple<const char*, const char*, const char*, const char*>, std::tuple<const char*, const char*>, const char*, const char*> __funcinfo__myKernel()

// TORCH:      m.def("myKernel", &myKernel, "myKernel");

// TORCH:      m.def("__funcinfo__myKernel", &__funcinfo__myKernel, "__funcinfo__myKernel");