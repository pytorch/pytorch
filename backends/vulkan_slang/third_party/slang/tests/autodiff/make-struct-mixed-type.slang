//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MixedType : IDifferentiable
{
    no_diff float noDiffField;
    float field;
}

[BackwardDifferentiable]
float f1(MixedType m)
{
    return 2.0 * m.field;
}

[BackwardDifferentiable]
float f(MixedType m)
{
    MixedType m1 = { m.noDiffField, m.field };
    return f1(m1);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    MixedType v = { 0, 2.0 };
    MixedType.Differential dv = { 0.0 };
    var p = diffPair(v, dv);
    __bwd_diff(f)(p, 1.0);
    outputBuffer[0] = p.d.field;
}