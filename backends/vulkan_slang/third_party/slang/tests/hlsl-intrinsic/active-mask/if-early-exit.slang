// if-early-exit.slang

// Test active mask synthesis for an `if` where one side
// unconditionally exits the current function/scope,
// and thus cannot "re-converge" after the conditional.

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name buffer
RWStructuredBuffer<int> buffer;

#define THREAD_COUNT 4
#define WRITE(IDX) buffer[IDX*THREAD_COUNT + tid] = WaveGetActiveMask()

void test(int tid)
{
	WRITE(0);
	if(tid & 1)
	{
		WRITE(1);
		return;
	}
	else
	{
		WRITE(2);
	}
	WRITE(3);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}