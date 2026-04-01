//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct AggType : IDifferentiable
{
    float field1;
    float field2;
}

[BackwardDifferentiable]
AggType makeTypeWithNonDiffOps(no_diff float f1, float f2)
{
    AggType v = { f1, f2 };
    return v;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    AggType v = { 0, 2.0 };
    AggType.Differential dv = { 1.3, 4.2 };

    var f2pair = diffPair(2.0, 0.0); 

    __bwd_diff(makeTypeWithNonDiffOps)(1.0, f2pair, dv);
    outputBuffer[0] = f2pair.d;
}