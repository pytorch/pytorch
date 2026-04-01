//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

struct Buggy<let N : int>
{
    float m(float x) { return N * x; }

    [BackwardDerivativeOf(m)]
    void mDiff(inout DifferentialPair<float> x, float dResult)
    {
        updateDiff(x, N * dResult);
    }
}

[Differentiable]
float test(float x)
{
    Buggy<2> b;
    return b.m(x);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    var a = diffPair(3.0);
    __bwd_diff(test)(a, 1.0);
    outputBuffer[dispatchThreadID.x] = a.d;
    // CHECK: 2.0
}
