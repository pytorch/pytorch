// rate-param.slang
//DISABLE_TEST:SIMPLE: -target hlsl -entry computeMain -stage compute 

//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

groupshared uint gs_values[4];

void someFunction(inout groupshared uint a[4], int index, int value)
{
    a[index] += value;
}

[shader("compute")]
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = (int)dispatchThreadID.x;
       
    // Initialize
    gs_values[3 - index] = index;   
       
    GroupMemoryBarrierWithGroupSync();
       
    someFunction(gs_values, index, index * 2 + 1);
       
    GroupMemoryBarrierWithGroupSync();
       
    outputBuffer[index] = gs_values[index];
}
