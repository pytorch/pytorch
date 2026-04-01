//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type

extension float
{
    float sum() { return this + 1; }
}

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride = 4)
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    float v = 10.0;
    // CHECK: 11
    outputBuffer[0] = v.sum();
}