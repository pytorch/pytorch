
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A : IDifferentiable
{
    float x;
    float y;
};

[BackwardDifferentiable]
A f(A a)
{
    // Read/writes to local struct variables won't be SSA'd out by default.
    // The backward diff preparation pass will kick in to create temp vars for them.
    A aout;
    aout.y = 2 * a.x;
    aout.y = aout.y + 2 * a.x;
    aout.x = aout.y + 5 * a.x;

    // The result should be equivalent to:
    /*
    A aout;
    var tmp = 2 * a.x;
    tmp = tmp + 2 * a.x;
    aout.y = tmp;
    aout.x = tmp + 5 * a.x;
    */
    return aout;

}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a = {1.0, 2.0};

    var dpa = diffPair(a);

    A.Differential dout = {1.0, 1.0};
    
    __bwd_diff(f)(dpa, dout);
    outputBuffer[0] = dpa.d.x; // Expect: 13
    outputBuffer[1] = dpa.d.y; // Expect: 0
}
