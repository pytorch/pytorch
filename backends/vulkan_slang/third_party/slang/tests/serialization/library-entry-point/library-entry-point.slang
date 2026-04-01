//TEST_IGNORE_FILE:

// libray-entry-point.slang
// Testing for the scenarion when an entry point is declared in a library

[__extern] int doSomething(int a);

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = int(dispatchThreadID.x);
    outputBuffer[index] = doSomething(index);
}

