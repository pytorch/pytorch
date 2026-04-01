//TEST(fxc):SIMPLE:-pass-through fxc -target dxbc -entry computeMain -stage compute -profile sm_5_1 

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
    // Error should be here... as gOutputBuffer is not defined...
	gOutputBuffer[tid] = dispatchThreadID.x * 0.5f;
}