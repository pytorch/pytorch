// Test calling differentiable function through dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IInterface
{
    static float calc(float x);
}

struct A : IInterface
{
    static float calc(float x) { return 1.0; }
};

struct B : IInterface
{
    static float calc(float x) { return 2.0; }
};

void dsqr<T:IInterface>(T obj, inout DifferentialPair<float> x, float dOut)
{
    float diff = 2.0 * x.p * dOut;
    updateDiff(x, diff);
}

[BackwardDerivative(dsqr)]
float sqr<T:IInterface>(T obj, float x)
{
    return no_diff(obj.calc(x)) + x * x;
}

// Use automatically differentiated outer function to triger the primal/propagate func generation logic
// on a function that has user provided backward derivative.
[BackwardDifferentiable]
float sqr_outter<T:IInterface>(T obj, float x)
{
    return sqr(obj, x);
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance B:IInterface = 1


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var obj = createDynamicObject<IInterface>(dispatchThreadID.x, 0); // A
    var p = DifferentialPair<float>(2.0, 1.0);
    __bwd_diff(sqr_outter)(obj, p, 1.0); // A.calc, expect 4
    outputBuffer[0] = p.d;

    obj = createDynamicObject<IInterface>(dispatchThreadID.x + 1, 0); // B
    p = DifferentialPair<float>(1.5, 1.0);
    __bwd_diff(sqr)(obj, p, 1.0);                                                     // A.calc, expect 4
    outputBuffer[1] = p.d; // B.calc, expect 3
}
