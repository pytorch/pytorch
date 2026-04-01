//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type

extension float2x2
{
    float sum() { return this[0][0] + this[1][1]; }
}

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride = 4)
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    float2x2 v = {1,2,3,4};
    // CHECK: 5
    outputBuffer[0] = v.sum();
}