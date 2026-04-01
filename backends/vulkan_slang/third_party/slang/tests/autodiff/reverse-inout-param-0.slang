//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
void g(float x, out float y)
{
    y = (x + 1) * (x + 1);
}

[BackwardDifferentiable]
void f(float x, out float y)
{
    g(x, y);
    g(x, y);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var x = diffPair(5.0);
    float yDiffOut = 1.0;

    __bwd_diff(f)(x, yDiffOut);

    outputBuffer[0] = x.p; // should be 5, since bwd_diff does not write back new primal val.
    outputBuffer[1] = x.d; // 12.0
}