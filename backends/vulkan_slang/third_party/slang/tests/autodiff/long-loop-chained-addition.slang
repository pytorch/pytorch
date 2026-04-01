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
    int i = 1;
    [MaxIters(30)]
    do
    {
        term *= -1.0f * x * x / ((2 * i) * (2 * i + 1));
        i += 2;
        i++;
        result += term;
        i -= 2;
    } while (i < iterations);
    return result;
}

// Check that the intermediate context of sin_series does not have an array for `i`.
// This test inparticular checks that can identify induction variables with
// more than one operation applied to them during the loop

// CHECK: struct s_bwd_prop_sin_series_Intermediates
// CHECK-NOT: int {{[A-Za-z0-9_]+}}[{{.*}}]
// CHECK: }

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    var x = diffPair(float.getPi(), 1.0);

    __bwd_diff(sin_series)(x, 30, 1.0f);

    outputBuffer[0] = x.d; // -1.0
}
