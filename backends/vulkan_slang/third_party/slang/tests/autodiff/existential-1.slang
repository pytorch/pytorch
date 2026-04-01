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
    int data1;

    __init(int data1) { this.data1 = data1; }

    [Differentiable]
    float calc(float x) { return x * x * x * data1; }
};

struct B : IInterface
{
    int data1;
    int data2;

    __init(int data1, int data2) { this.data1 = data1; this.data2 = data2; }

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
        obj = no_diff(A(2));
    else
        obj = no_diff(B(2, 3));

    return doThing(obj, x);
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance B:IInterface = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = fwd_diff(f)(dispatchThreadID.x, DifferentialPair<float>(1.0, 2.0)).d; // A.calc, expect 12
    outputBuffer[1] = fwd_diff(f)(dispatchThreadID.x + 1, DifferentialPair<float>(1.5, 1.0)).d; // B.calc, expect 18

    {
        var dpx = diffPair(1.0);
        bwd_diff(f)(dispatchThreadID.x, dpx, 2.0); // A.calc, expect 12
        outputBuffer[2] = dpx.d;
    }

    {
        var dpx = diffPair(1.5);
        bwd_diff(f)(dispatchThreadID.x + 1, dpx, 1.0); // B.calc, expect 18
        outputBuffer[3] = dpx.d;
    }
}
