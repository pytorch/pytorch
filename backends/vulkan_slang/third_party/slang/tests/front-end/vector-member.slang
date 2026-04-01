//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type

extension float3
{
    float sum() { return this.x + this.y + this.z; }
}

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride = 4)
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    float3 v = { 1, 2, 3 };
    // CHECK: 6
    outputBuffer[0] = v.sum();
}