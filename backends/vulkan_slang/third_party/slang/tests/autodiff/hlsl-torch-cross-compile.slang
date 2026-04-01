//TEST:SIMPLE(filecheck=HLSL): -target hlsl -line-directive-mode none -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none
//TEST:SIMPLE(filecheck=TORCH): -target torch -line-directive-mode none

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[Differentiable]
float func1(float x)
{
    return x * 4;
}

[AutoPyBindCUDA]
[CUDAKernel]
void torchMain(TensorView<float> v)
{
    v[0] = func1(v[0]);
    v[1] = func1(v[1]);
}

// Shouldn't see torchMain (or its transformations) anywhere in the HLSL output
// HLSL-NOT:torchMain
// HLSL:func1
// HLSL-NOT:torchMain
// HLSL:computeMain
// HLSL-NOT:torchMain

[Differentiable]
float func2(float a)
{
    return a;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat dpb = dpfloat(1.5, 1.0);

        outputBuffer[0] = fwd_diff(func1)(dpa).d;                 // Expect: 1
        outputBuffer[1] = fwd_diff(func2)(dpfloat(dpa.p, 0.0)).d;   // Expect: 0
    }
}

// Ensure that the generated CUDA and Torch kernels do have torchMain & its transformations

// TORCH:      {{^SLANG_PRELUDE_EXPORT$}}
// TORCH-NEXT: void __kernel__torchMain(TensorView {{[[:alnum:]_]+}});

// CUDA: __global__ void __kernel__torchMain(TensorView {{[[:alnum:]_]+}})
