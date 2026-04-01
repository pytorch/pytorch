//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK: [branch]
// CHECK: [flatten]

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    
    int v = idx;
    
    [branch]
    if (idx != 0)
    {
        ++v;
    }
    
    [flatten]
    if (idx > 2)
    {
        v += v;
    }

	outputBuffer[dispatchThreadID.x] = v;
}