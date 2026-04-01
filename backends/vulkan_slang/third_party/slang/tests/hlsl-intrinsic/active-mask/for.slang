// for.slang

// Test active mask synthesis for a standard `for`
// loop over an integer, with no `break` or `continue`
// logic.

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name buffer
RWStructuredBuffer<int> buffer;

#define THREAD_COUNT 4
#define WRITE(IDX) buffer[IDX*THREAD_COUNT + tid] = WaveGetActiveMask()

void test(int tid)
{
	for(int ii = 0; ii < tid; ++ii)
	{
		WRITE(ii);
	}
	WRITE(THREAD_COUNT);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}