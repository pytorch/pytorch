// buffer-barrier-test.slang

[shader("compute")]
[numthreads(4,1,1)]
void computeA(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> inBuffer,
    uniform RWStructuredBuffer<float> outBuffer)
{
    outBuffer[sv_dispatchThreadID.x] = inBuffer[sv_dispatchThreadID.x];
}

[shader("compute")]
[numthreads(4,1,1)]
void computeB(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform StructuredBuffer<float> inBuffer,
    uniform RWStructuredBuffer<float> outBuffer)
{
    outBuffer[sv_dispatchThreadID.x] = inBuffer[sv_dispatchThreadID.x] + 10.0f;
}

