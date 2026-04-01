// shader.slang

//TEST_INPUT:ubuffer(random(float, 4096, -1.0, 1.0), stride=4):name=ioBuffer
RWStructuredBuffer<float> ioBuffer;

[shader("compute")]
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    float i = ioBuffer[tid];
    float o = i < 0.5 ? (i + i) : sqrt(i);

    ioBuffer[tid] = o;
}
