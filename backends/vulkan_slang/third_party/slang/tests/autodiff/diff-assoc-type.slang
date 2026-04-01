// Tests automatic synthesis of Differential type and method requirements.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct B : IDifferentiable
{
    float x;
}

float myFunc(DifferentialPair<B.Differential> d)
{
    return d.p.x + d.d.x;
}

float myFunc2(DifferentialPair<B>.Differential d)
{
    return d.p.x + d.d.x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    B.Differential bd;
    bd.x = 1.0;
    outputBuffer[0] = myFunc(diffPair(bd, bd)) + myFunc2(diffPair(bd, bd));
}
