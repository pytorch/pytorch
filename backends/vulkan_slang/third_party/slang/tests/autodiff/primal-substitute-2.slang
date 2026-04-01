//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

float original(float x)
{
    return x * x;
}

[PrimalSubstituteOf(original)]
[BackwardDifferentiable]
float primalSubst(float x)
{
    return 2.0f * x * x;
}

[BackwardDifferentiable]
float caller(float x)
{
    return original(x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var a = diffPair(3.0, 1.0);
    __bwd_diff(caller)(a, 1.0);
    outputBuffer[0] = a.d;                                      // Expect: 12.0
    outputBuffer[1] = __fwd_diff(caller)(diffPair(3.0, 1.0)).p; // Expect: 18.0
    outputBuffer[2] = caller(3.0); // Expect: 9.0
}
