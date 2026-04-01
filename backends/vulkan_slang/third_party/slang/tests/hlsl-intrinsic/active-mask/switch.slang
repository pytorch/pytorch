// switch.slang

// Test active mask synthesis for a trivial `switch` statement

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -xslang -DHACK -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -xslang -DHACK -shaderobj

// Note: this test is currently disabled on the CUDA
// target because we do not synthesize the active
// mask value we want/expect to see.
//
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name buffer
RWStructuredBuffer<int> buffer;

#define THREAD_COUNT 4
#define LOC_COUNT 5
#define WRITE(LOC) buffer[tid + (LOC)*THREAD_COUNT] = 0xA0000000 | (tid << 24) | (LOC << 8) | int(WaveGetActiveMask())

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

	default:
		WRITE(3);
		break;
	}
	WRITE(4);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}