// Test calling differentiable function through dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IInterface
{
    [ForwardDifferentiable]
    static float calc(float x);
}

struct A : IInterface
{
    [ForwardDifferentiable]
    static float calc(float x) { return x * x * x; }
};

struct B : IInterface
{
    [ForwardDifferentiable]
    static float calc(float x) { return x * x; }
};

[ForwardDifferentiable]
float sqr<T:IInterface>(T obj, float x)
{
    return obj.calc(x);
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance B:IInterface = 1


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var obj = createDynamicObject<IInterface>(dispatchThreadID.x, 0); // A
    outputBuffer[0] = __fwd_diff(sqr)(obj, DifferentialPair<float>(2.0, 1.0)).d; // A.calc, expect 12

    obj = createDynamicObject<IInterface>(dispatchThreadID.x + 1, 0); // B
    outputBuffer[1] = __fwd_diff(sqr)(obj, DifferentialPair<float>(1.5, 1.0)).d; // B.calc, expect 3
}
