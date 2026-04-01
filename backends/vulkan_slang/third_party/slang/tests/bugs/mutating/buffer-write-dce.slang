//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

// Confirm that calling a mutating method to write to a buffer location doesn't
// get DCE'd.

struct C
{
    int a;
    [mutating]
    void set(int val)
    {
        a = val;
    }
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out, name outputBuffer
RWStructuredBuffer<C> outputBuffer;
void writeOutput(int tid)
{
    outputBuffer[tid].set(1);
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    writeOutput(tid);
}
