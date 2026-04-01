// Tests automatic synthesis of Differential type and method requirements.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct B : IDifferentiable
{
    float x;
}

struct A : IDifferentiable
{
    B b;
    float y;
};

typedef DifferentialPair<A> dpA;

A nonDiff(A a)
{
    return a;
}

[ForwardDifferentiable]
A f(A a)
{
    A aout;
    aout.y = 2 * a.b.x;
    aout.b.x = 5 * a.b.x;

    return no_diff(nonDiff(aout));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        A a = {1.0, 2.0};
        A.Differential b = {0.2};
        dpA dpa = dpA(a, b);
        outputBuffer[0] = __fwd_diff(f)(dpa).d.b.x;         // Expect: 0
        outputBuffer[1] = A.dadd(b, b).b.x;                 // Expect: 0.4
        outputBuffer[2] = A.dmul<float>(2.0, b).b.x;        // Expect: 0.4
    }
} 
