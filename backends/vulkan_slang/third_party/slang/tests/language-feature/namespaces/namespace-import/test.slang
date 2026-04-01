//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

module test;

import m;

namespace ns1.ns2
{
    int f()
    {
        // Should be able to access g() from m.
        return g();
    }
}

using ns1.ns2;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = f() + g() + ns1.ns2.g();
    outputBuffer[tid] = outVal;
    // CHECK: 3
}
