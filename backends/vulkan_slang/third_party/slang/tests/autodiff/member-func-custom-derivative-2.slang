//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo
{
    [BackwardDifferentiable]
    float3 test(float v, uint offset);
}
struct A : IFoo
{
    float x;

    float3 f(float v, uint offset)
    {
        return v * v;
    }

    // Provide a backward diff, but leave out forward diff.
    [BackwardDerivativeOf(f)]
    [TreatAsDifferentiable]
    void diff_f(inout DifferentialPair<float> v, uint offset, float3 dOut)
    {
        v = diffPair(v.p, 2 * v.p * dOut.x);
    }

    [BackwardDifferentiable]
    float3 test(float v, uint offset)
    {
        return f(v, 0);
    }
}

[BackwardDifferentiable]
float3 test(IFoo obj, float v)
{
    return obj.test(v, 0);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A a = {0.0};
    var p = diffPair(3.0, 0.0);
    let rs = __bwd_diff(test)(a, p, 1.0);
    outputBuffer[0] = p.d;
}
