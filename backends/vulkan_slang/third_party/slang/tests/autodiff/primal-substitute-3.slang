//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo
{
    float doSomething();
}

struct A : IFoo
{
    float doSomething()
    {
        return 0.0f;
    }
}

// A normal function that calls doSomething().
float original<T : IFoo>(T p, float x)
{
    p.doSomething();
    return x * x;
}

[PrimalSubstituteOf(original)]
[BackwardDifferentiable]
float primalSubst<T : IFoo>(T p, float x)
{
    return 2.0f * x * x;
}

[BackwardDifferentiable]
float caller(IFoo d, float x)
{
    return original(d, x);
}

//TEST_INPUT: type_conformance A:IFoo = 0

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var obj = createDynamicObject<IFoo>(dispatchThreadID.x, 0); // A

    var a = diffPair(3.0, 1.0);
    __bwd_diff(caller)(obj, a, 1.0);
    outputBuffer[0] = a.d;                                         // Expect: 12.0
    outputBuffer[1] = __fwd_diff(caller)(obj, diffPair(3.0, 1.0)).p; // Expect: 18.0
    outputBuffer[2] = caller(obj, 3.0); // Expect: 9.0
}
