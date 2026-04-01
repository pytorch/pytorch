//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
bool conditionFunc(no_diff float a, inout float x)
{
    x = x * a;
    return x > 100.f;
}

[BackwardDifferentiable]
float outerFunc(no_diff float a, float x)
{
    if (conditionFunc(a, x))
        return x;
    else
        return -x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float a = 10.0;
    DifferentialPair<float> dpx = DifferentialPair<float>(4.f, 1.f);
    __bwd_diff(outerFunc)(a, dpx, 1.0);

    outputBuffer[0] = dpx.d;
}