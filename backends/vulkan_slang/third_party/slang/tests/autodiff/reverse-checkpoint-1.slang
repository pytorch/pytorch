//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none
//TEST:SIMPLE(filecheck=CHK):-target glsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float> dpfloat;
typedef float.Differential dfloat;

[BackwardDifferentiable]
[PreferCheckpoint]
float g(float x)
{
    return log(x);
}

//CHK: note: checkpointing context of 4 bytes associated with function: 'f'
[BackwardDifferentiable]
float f(int p, float x)
{
    float y = 1.0;
    // Test that phi parameter can be restored.
    if (p == 0)
        //CHK: note: 4 bytes (float) used to checkpoint the following item:
        y = g(x);

    return y * y;
}

// Check that there are no calls to primal_g in bwd_f.

// CHECK: void s_bwd_f_{{[0-9]+}}
// CHECK-NOT: {{[_a-zA-Z0-9]+}} = s_primal_ctx_g_{{[0-9]+}}
// CHECK: return


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    dpfloat dpa = dpfloat(2.0, 0.0);

    __bwd_diff(f)(0, dpa, 1.0f);
    outputBuffer[0] = dpa.d; // Expect: 1
}

//CHK-NOT: note