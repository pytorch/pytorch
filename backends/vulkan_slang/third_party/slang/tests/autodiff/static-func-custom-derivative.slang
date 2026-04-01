//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A : IDifferentiable
{
    float x;

    [ForwardDerivative(diff_f)]
    static float f(float v)
    {
        return v * v;
    }

    static DifferentialPair<float> diff_f(DifferentialPair<float> v)
    {
        return diffPair(v.p * v.p, v.p * v.d * 2.0);
    }

    [BackwardDerivative(diff_g)]
    static float g(float v)
    {
        return v * v;
    }

    static void diff_g(inout DifferentialPair<float> v, float.Differential dOut)
    {
        v = diffPair(v.p, dOut * 2.0);
    }
}

[ForwardDifferentiable]
float test(A obj, float v)
{
    return obj.f(v);
}

[BackwardDifferentiable]
float test2(A obj, float v)
{
    return obj.g(v);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a = {0.0};
    var p = diffPair(3.0, 1.0);
    let rs = fwd_diff(test)(diffPair(a, {1.0}), p);

    var q = diffPair(3.0);
    var qa = diffPair(a);
    bwd_diff(test2)(qa, q, 1.0);

    outputBuffer[0] = rs.d;
    outputBuffer[1] = q.d;
}
