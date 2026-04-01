//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A
{
    float x;

    [ForwardDerivative(diff_f)]
    float f(float v)
    {
        return v * v;
    }

    DifferentialPair<float> diff_f(DifferentialPair<float> v)
    {
        return diffPair(v.p * v.p, v.p * v.d * 2.0);
    }
}

[ForwardDifferentiable]
float test(A obj, float v)
{
    return obj.f(v);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a = {0.0};
    var p = diffPair(3.0, 1.0);
    let rs = __fwd_diff(test)(a, p);
    outputBuffer[0] = rs.d;
}
