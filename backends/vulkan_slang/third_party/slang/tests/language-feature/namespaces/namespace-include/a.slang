//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

module a;

__include b;

namespace ns1
{
namespace ns2
{
    int l() { return k(); } // should be able to find k() in b.
}
}

using ns1.ns2;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = k() + l() + ns1.ns2.f();
    outputBuffer[tid] = outVal;
    // CHECK: 7
}
