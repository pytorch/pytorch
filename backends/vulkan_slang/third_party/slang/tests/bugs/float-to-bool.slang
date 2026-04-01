//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -xslang -Wno-30081
//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -xslang -Wno-30081

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0 0], stride = 4)
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT: set inputBuffer = ubuffer(data=[0.0 0.2 1.0 1.2], stride = 4)
RWStructuredBuffer<float4> inputBuffer;
[numthreads(1,1,1)]
void computeMain()
{
    bool4 bv = inputBuffer[0];
    outputBuffer[0] = bv.x;
    outputBuffer[1] = bv.y;
    outputBuffer[2] = bv.z;
    outputBuffer[3] = bv.w;
    bool b = inputBuffer[0].y;
    outputBuffer[4] = b;
}

// CHECK: 0
// CHECK: 1
// CHECK: 1
// CHECK: 1
// CHECK: 1