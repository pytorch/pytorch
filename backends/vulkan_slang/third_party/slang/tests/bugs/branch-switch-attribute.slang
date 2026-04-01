//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK: [branch]

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    
    int v = idx;
   
   [branch]
    switch (idx)
    {
        case 0: v ++; break;
        case 2: v -= 7; break;
        default: v--;
    }

	outputBuffer[dispatchThreadID.x] = v;
}