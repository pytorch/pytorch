//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

module b;

import a;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = f() + ns.f();
    outputBuffer[tid] = outVal;
    // CHECK: 2
}
