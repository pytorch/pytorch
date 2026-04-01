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

[BackwardDifferentiable]
void g(D d, int i, out D outArray[1][2])
{
    outArray[0][0].m = d.n;
    outArray[0][0].n = d.m * 3.0f + 1.0;

    outArray[0][i].m = d.n * d.n;
    outArray[0][i].n = d.m * 4.0f + 1.0;
}

[BackwardDifferentiable]
void f(D d, int i, out D outArray[1][2])
{
    g(d, 0, outArray);
    g(d, i, outArray);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    D d = { 1.0, 2.0 };

    var dpd = diffPair(d);

    D.Differential tmp[1][2] = { { { 1.0, 1.0 }, { 1.0, 1.0 } } };
    __bwd_diff(f)(dpd, 1, tmp);

    outputBuffer[0] = dpd.d.m;
    outputBuffer[1] = dpd.d.n;
}