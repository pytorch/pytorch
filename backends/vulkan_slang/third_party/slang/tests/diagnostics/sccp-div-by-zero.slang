//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -profile cs_5_0 -target hlsl
RWStructuredBuffer<uint> outputBuffer;

// CHECK: divide by zero
uint check<let b : bool>()
{
    return 1 / int(b);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
    
    uint a = check<false>();
    uint b = check<true>();
    
	outputBuffer[tid] = a + b; 
}