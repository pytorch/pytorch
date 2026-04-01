// Test calling dynamic dispatched generic function from differentiable function.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo
{
    float f();
}

interface IInterface
{
    [BackwardDifferentiable]
    float calc<T:IFoo>(T t, float x);
}

struct A : IFoo
{
    float f() { return 5.0; }
};

struct B : IInterface
{
    [BackwardDifferentiable]
    float calc<T : IFoo>(T t, float x)
    {
        return t.f() * x;
    }
};

[BackwardDifferentiable]
float test(IFoo foo, IInterface obj, float x)
{
    return obj.calc(foo, x) * x;
}

//TEST_INPUT: type_conformance A:IFoo = 0
//TEST_INPUT: type_conformance B:IInterface = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var obj = createDynamicObject<IInterface>(dispatchThreadID.x, 1); // B
    var foo = createDynamicObject<IFoo>(0, 0); // A
    var p = diffPair(3.0);
    __bwd_diff(test)(foo, obj, p, 1.0);
    outputBuffer[0] = p.d;
}
