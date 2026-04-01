// Test calling differentiable generic function with inout generic parameter.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IInterface
{
    [mutating]
    float sample();
}

struct A : IInterface
{
    float z;
    [mutating]
    float sample() { z = z + 1.0; return 1.0; }
};

[ForwardDifferentiable]
float sqr<T:IInterface>(inout T obj, float x)
{
    return (no_diff obj.sample()) + x*x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A obj;
    obj.z = 0.0;
    outputBuffer[0] = __fwd_diff(sqr)(obj, DifferentialPair<float>(2.0, 1.0)).d;
}
