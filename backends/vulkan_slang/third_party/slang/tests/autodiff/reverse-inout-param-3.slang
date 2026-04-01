//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
void assign(inout float rs, float v)
{
    rs = v;
}

[BackwardDifferentiable]
void f(inout float p, float x)
{
    assign(p, x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var x = diffPair(5.0, 0.0);
    var pp = diffPair(1.0, 3.0);
    __bwd_diff(f)(pp, x);

    outputBuffer[0] = pp.p; // should be 1, since bwd_diff does not write back new primal val.
    outputBuffer[1] = x.d; // 3
}