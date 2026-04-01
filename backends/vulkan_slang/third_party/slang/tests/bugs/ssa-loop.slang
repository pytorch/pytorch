// ssa-loop.slang

// Bug related to SSA form for loops

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

int test(int val)
{
    int N = val;
    int x = 0;
    int y = 1;
    for(int i = 0; i < N; ++i)
    {
        int t = x;
        x = y;
        y = t;
    }
    return x*16 + y;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inputVal = tid;
    int outputVal = test(inputVal);
    gOutputBuffer[tid] = outputVal;
}
