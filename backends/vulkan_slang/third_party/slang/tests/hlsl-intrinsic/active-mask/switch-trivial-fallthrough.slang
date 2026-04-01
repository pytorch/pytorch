// switch-trivial-fallthrough.slang

// Test active mask synthesis for a `switch` statement
// that exhibits "trivial fall-through" from one `case`
// to another.

// We've ended up having to disable this test everywhere,
// which really speaks to the way that the "active mask" concept
// in D3D and Vulkan was never really specified to a level
// that makes it meaningful.
//
// Even if the specs get fixed at some point down the line,
// we can't really re-enable these tests because they'd fail on way
// too many GPUs and driver versions.

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -xslang -DHACK -shaderobj
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -xslang -DHACK -shaderobj

// Note: this test is currently disabled on the CUDA
// target because we do not synthesize the active
// mask value we want/expect to see.
//
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0

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
	case 1:
		WRITE(0);
		break;

	case 2:
	default:
		WRITE(2);
		break;
	}
	WRITE(4);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}