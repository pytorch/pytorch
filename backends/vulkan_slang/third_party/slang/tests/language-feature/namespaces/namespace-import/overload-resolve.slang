//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

module test;

namespace test
{
    vector<uint, N> f<int N>(vector<uint, N> x)
    {
        return x;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    // Should be able to resolve 'test' properly (and not get confused by the module declaration with the same name)
    int outVal = test.f<3>(vector<uint, 3>(1, 2, 3)).y;
    outputBuffer[tid] = outVal;
    // CHECK: 2
}
