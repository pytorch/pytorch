
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A : IDifferentiable
{
    float x;
    float y;
};

typedef DifferentialPair<A> dpA;

[BackwardDifferentiable]
A f(A a)
{
    A aout;
    aout.y = 2 * a.x;
    aout.x = 5 * a.x;

    return aout;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        A a = {1.0, 2.0};
        A.Differential azero = {0.0, 0.0};

        dpA dpa = dpA(a, azero);

        A.Differential dout = {1.0, 1.0};
        
        __bwd_diff(f)(dpa, dout);
        outputBuffer[0] = dpa.d.x; // Expect: 7
        outputBuffer[1] = dpa.d.y; // Expect: 0
    }
}
