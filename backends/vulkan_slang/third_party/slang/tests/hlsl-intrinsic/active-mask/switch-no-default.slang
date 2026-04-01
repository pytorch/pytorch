// switch-no-default.slang

// Test active mask synthesis for a `switch` statement
// with no `default` case (such that unhandled values
// branch directly to the `break` target after the
// `switch`).

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -xslang -DHACK
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -xslang -DHACK
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name buffer
RWStructuredBuffer<int> buffer;

#define THREAD_COUNT 4
#define LOC_COUNT 5
#define WRITE(LOC) buffer[tid + (LOC)*THREAD_COUNT] = 0xA0000000 | (tid << 24) | (LOC << 8) | WaveGetActiveMask()

//TEST_INPUT:cbuffer(data=[0 1]):name C
cbuffer C
{
	int alwaysFalse;
	int alwaysTrue;
}

void test(int tid)
{
	switch(tid)
	{
	case 0:
		WRITE(0);
		break;

	case 1:
		WRITE(1);
		break;

	case 2:
		WRITE(2);
		break;

	// NOTE: no `default:`
	}
	WRITE(4);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}