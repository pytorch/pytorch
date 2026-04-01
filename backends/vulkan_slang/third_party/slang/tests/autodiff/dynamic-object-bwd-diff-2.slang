// Test calling backward differentiable function through dynamic dispatch, where the interface
// being dispatched inherits from IDifferentiable, so that `this` is differentiable.

//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IInterface : IDifferentiable
{
    [BackwardDifferentiable]
    float calc(float x);
}

struct C : IInterface
{
    [BackwardDifferentiable]
    float calc(float x) { return 2 * x; }
}

struct A : IInterface
{
    float a;
    [BackwardDifferentiable]
    float calc(float x)
    {
        return a * x * x;
    }
};


[BackwardDifferentiable]
float run(int id, float x, no_diff float y)
{
    IInterface obj = createDynamicObject<IInterface>(id, y);
    C c = {};
    return obj.calc(x);
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance C:IInterface = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        var p = diffPair(3.0);

        __bwd_diff(run)(0, p, 0.5, 1.0f);
        outputBuffer[0] = p.d; // A.calc, expect 3
    }

    {
        var p = diffPair(3.0);

        __bwd_diff(run)(1, p, 1.5, 1.0f);
        outputBuffer[1] = p.d; // c.calc, expect 2
    }
}
