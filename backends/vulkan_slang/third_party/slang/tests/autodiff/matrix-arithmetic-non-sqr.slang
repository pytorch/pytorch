//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[Differentiable]
float h_outer(float a, float b, float c)
{
    const float3x2 m2 = float3x2(2 * a, 0.0,
        0.0, 3 * b,
        0.0, 5 * c);

    const float3x3 m1 = float3x3(1.f);
    return h(m1, m2);
}

[Differentiable]
float h(float3x3 x, float3x2 y)
{
    let res = mul(x, y);
    return dot(mul(res, float2(1.0)), float3(1.0));
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Do a bwd_diff test for h_outer
    var dpa = diffPair(1.f, 0.f);
    var dpb = diffPair(1.f, 0.f);
    var dpc = diffPair(1.f, 0.f);
    bwd_diff(h_outer)(dpa, dpb, dpc, 1.f);

    outputBuffer[1] = dpa.d;
    outputBuffer[2] = dpb.d;
    outputBuffer[3] = dpc.d;

    // CHECK: type: float
    // CHECK: 6.0
    // CHECK: 9.0
    // CHECK: 15.0
}
