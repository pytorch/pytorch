//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
float sin_series(float x, int iterations)
{
    float result = x;
    float term = x;
    [MaxIters(35)]
    for (int i = 1; i < iterations; i++)
    {
        if(i == 32)
            i += 1;
        term *= -1.0f * x * x / ((2 * i) * (2 * i + 1));
        result += term;
    }
    return result;
}

// Check that the intermediate context of sin_series still has an array for
// `i`. This test checks that the induction variable finder doesn't
// accidentally succeed all the time

// CHECK: struct s_bwd_prop_sin_series_Intermediates
// CHECK: int {{[A-Za-z0-9_]+}}[{{.*}}]
// CHECK: }

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    var x = diffPair(float.getPi(), 1.0);

    __bwd_diff(sin_series)(x, 30, 1.0f);

    outputBuffer[0] = x.d; // -1.0
}
