//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct D : IDifferentiable
{
    float n;
    float m;
}

[PreferRecompute]
[BackwardDifferentiable]
void g(no_diff float p, inout float x)
{
    x = p * ((x+1)*(x+1));
}

[PreferRecompute]
[BackwardDifferentiable]
void f(no_diff float p, inout float x)
{
    g(p, x);
    g(p, x);
}
[BackwardDifferentiable]
float f_ref(no_diff float p, float x)
{
    float y1 = p * (x+1)*(x+1);
    float y2 = p * (y1+1)*(y1+1);
    return y2;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var x = diffPair(2.0, 1.0);

    __bwd_diff(f)(3.0, x);

    outputBuffer[0] = x.p; // should be 2, since bwd_diff does not write back new primal val.
    outputBuffer[1] = x.d; // 3024

    var refVal = __fwd_diff(f_ref)(3.0, diffPair(2.0, 1.0)).d;
    outputBuffer[2] = refVal; // 3024

}
