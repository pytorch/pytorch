//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct D : IDifferentiable
{
    float n;
    float m;
}

[ForwardDifferentiable]
float f(D d, int i, out D outArray[1][2])
{
    outArray[0][0].m = d.n;
    outArray[0][0].n = d.m * 3.0f + 1.0;

    outArray[0][1].m = d.n * d.n;
    outArray[0][1].n = d.m * 4.0f + 1.0;

    return outArray[0][0].m + outArray[0][1].n * 2.0;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    D d = { 1.0, 2.0 };
    D.Differential dd = { 1.0, 1.0 };

    var dpd = diffPair(d, dd);

    D tmp[1][2] = { { { 1.0, 1.0 }, { 1.0, 1.0 } } };
    DifferentialPair<D[2][1]> dOut = diffPair(tmp);
    var result = __fwd_diff(f)(dpd, 1, dOut).d;

    // dOut.d has type D[2][1].Differential. Verify that it can coerce to D.Differential[2][1].
    D.Differential darrCopy[1][2] = dOut.d;

    outputBuffer[0] = result;
    outputBuffer[1] = dOut.d[0][0].n;
    outputBuffer[2] = dOut.d[0][0].m;
    outputBuffer[3] = darrCopy[0][1].n;
    outputBuffer[4] = darrCopy[0][1].m;
    
}
