//TEST:LANG_SERVER:
RWStructuredBuffer<float> outputBuffer;

[ForwardDifferentiable]
float f(float x)
{
    return x * x;
}

[ForwardDifferentiable]
float df(float x)
{
//HOVER:14,17
    return __fwd_diff();
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = __fwd_diff(df)(DifferentialPair<float>(x, 1.0)).d(); // Expect: 2.0
}
