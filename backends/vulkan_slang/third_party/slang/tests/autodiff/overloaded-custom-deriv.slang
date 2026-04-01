//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[ForwardDerivative(diff_f)]
float f(float v)
{
    return v * v;
}

DifferentialPair<float> diff_f(DifferentialPair<float> v)
{
    return diffPair(v.p * v.p, 2 * v.d * v.p);
}

[ForwardDerivative(diff_f)]
float2 f(float2 v)
{
    return v * v;
}

DifferentialPair<float2> diff_f(DifferentialPair<float2> v)
{
    return diffPair(v.p * v.p, float2(2 * v.d.x * v.p.x, 0.0));
}

[ForwardDifferentiable]
float test(float v)
{
    return f(v) + f(float2(v, v)).x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var p = diffPair(3.0, 1.0);
    let rs = __fwd_diff(test)(p);
    outputBuffer[0] = rs.d;
}
