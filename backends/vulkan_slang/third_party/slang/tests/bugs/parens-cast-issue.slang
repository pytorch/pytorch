//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
    int idx = (int)tid;
    
    float q = (idx * 0.5);
 
    int x = (uint)(int)q;
    
    // This combination of casts and parenthesis used to cause issues. 
    int z = ((uint)(int)q);
    
 	outputBuffer[tid] = z + x;
}