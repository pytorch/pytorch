// Test calling differentiable function through dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IInterface : IDifferentiable
{
    [Differentiable]
    float calc(float x);
}

struct A : IInterface
{
    float data1;

    [Differentiable]
    __init(float data1) { this.data1 = data1; }

    [Differentiable]
    float calc(float x) { return x * x * x * data1; }
};

struct B : IInterface
{
    float data1;
    float data2;

    [Differentiable]
    __init(float data1, float data2) { this.data1 = data1; this.data2 = data2; }

    [Differentiable]
    float calc(float x) { return x * x * data1 * data2; }
};

[Differentiable]
float doThing(IInterface obj, float x)
{
    return obj.calc(x);
}
 
[Differentiable]
float f(uint id, float x)
{
    IInterface obj;

    if (id == 0)
        obj = A(x); // x^4
    else
        obj = B(x, x); // x^4

    return doThing(obj, x) + doThing(obj, x); // 2 * x^4
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance B:IInterface = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        var dpx = diffPair(1.0);
        bwd_diff(f)(dispatchThreadID.x, dpx, 2.0); 
        outputBuffer[0] = dpx.d; // Expect: 2 * 4 * x^3 * dx = 8 * x^3 * dx = 16
    }

    {
        var dpx = diffPair(1.5);
        bwd_diff(f)(dispatchThreadID.x + 1, dpx, 1.0);
        outputBuffer[1] = dpx.d; // Expect: 2 * 4 * x^3 * dx = 8 * (1.5)^3 * 1.0 = 27
    }
}
