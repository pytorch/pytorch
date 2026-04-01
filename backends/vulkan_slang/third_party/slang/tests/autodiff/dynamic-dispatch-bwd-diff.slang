// Test calling backward differentiable function through dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IInterface
{
    [BackwardDifferentiable]
    float calc(float x);
}

struct A : IInterface
{
    float a;
    [BackwardDifferentiable]
    float calc(float x) { return a*x*x; }
};

struct B : IInterface
{
    float a;
    [BackwardDifferentiable]
    float calc(float x) { return a*x*x*x; }
};

[BackwardDifferentiable]
float run(IInterface obj, float x)
{
    return obj.calc(x);
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance B:IInterface = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var obj = createDynamicObject<IInterface>(dispatchThreadID.x, 0.5f); // A
    var p = diffPair(3.0);

    __bwd_diff(run)(obj, p, 1.0f);
    outputBuffer[0] = p.d; // A.calc, expect 3

    obj = createDynamicObject<IInterface>(dispatchThreadID.x + 1, 1.5f); // B
    p = diffPair(3.0);
    __bwd_diff(run)(obj, p, 1.0f);
    outputBuffer[1] = p.d; // B.calc, expect 40.5
}
