//DISABLE_TEST(smoke):SIMPLE: -target ptx -entry computeMain -stage compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cpu -compute 
//TEST(compute):COMPARE_COMPUTE:-cuda -compute 

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer : register(u0);

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = int(dispatchThreadID.x);
    
    outputBuffer[tid * 4] = tid;
    outputBuffer[tid * 4 + 1] = tid + 1;
    outputBuffer[tid * 4 + 2] = tid + 2; 
    outputBuffer[tid * 4 + 3] = tid + 3;
}
