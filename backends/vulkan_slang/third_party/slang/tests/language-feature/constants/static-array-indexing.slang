//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): 

int check<int v>()
{
    return v;
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    const int a[] = { 1, 2, 3, 4 };
    // CHECK: 4
    outputBuffer[tid] = check<a[3]>();
}
