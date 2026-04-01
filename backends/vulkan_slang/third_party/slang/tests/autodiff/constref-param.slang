//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cuda -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct NonDiff
{
    float a;
}

[Differentiable]
float myFunc(__constref NonDiff fIn, float x, __constref no_diff float y)
{
    return x * fIn.a + y;
}

// test that the ordering of no_diff and __constref doesn't matter.
[Differentiable]
float myFunc2(__constref NonDiff fIn, float x, no_diff __constref float y)
{
    return x * fIn.a + y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float a = 10.0;
    NonDiff fIn = { a };
    DifferentialPair<float> dpx = DifferentialPair<float>(4.f, 1.f);
    float rs = __fwd_diff(myFunc)(fIn, dpx, 1.0).d;
    float rs2 = __fwd_diff(myFunc2)(fIn, dpx, 1.0).d;

    // CHECK: 10.0
    // CHECK: 10.0

    outputBuffer[0] = rs;
    outputBuffer[1] = rs2;
}
