//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct NonDiff
{
    float a;
}

[ForwardDifferentiable]
bool myFunc(NonDiff fIn, inout float x)
{
    x = pow(x, fIn.a);
    return x > 100.f;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float a = 10.0;
    NonDiff fIn = { a };
    DifferentialPair<float> dpx = DifferentialPair<float>(4.f, 1.f);
    bool res = __fwd_diff(myFunc)(fIn, dpx);

    outputBuffer[0] = res?1:0;
}