// Test calling differentiable function through dynamic dispatch.

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

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
float doThing(IInterface obj, IInterface obj2, float x)
{
    return obj.calc(x);
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance B:IInterface = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        IInterface obj;
        if (dispatchThreadID.x == 0)
            obj = A(2);
        else
            obj = B(2, 3);

        IInterface obj2 = A(3);

        var dpx = diffPair(1.0);
        DifferentialPair<IInterface> dpobj = diffPair<IInterface>(obj);
        DifferentialPair<IInterface> dpobj2 = diffPair<IInterface>(obj2);

        bwd_diff(doThing)(dpobj, dpobj2, dpx, 2.0); // A.calc, expect 12

        IDifferentiable objd = dpobj.d;

        outputBuffer[0] = (isNullDifferential((objd)) ? 1.f : 0.f);
    }
}
