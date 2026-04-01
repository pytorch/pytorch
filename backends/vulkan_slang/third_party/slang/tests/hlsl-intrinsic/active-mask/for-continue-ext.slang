// for-continue-ext.slang

// Test case of a `for` loop that
// has multiple paths to continue
// the loop (both the ordinary one
// and an explicit `continue`)

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -xslang -DHACK
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -xslang -DHACK
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name buffer
RWStructuredBuffer<int> buffer;

#define THREAD_COUNT 4
#define LOC_COUNT 4
#define ITER_COUNT THREAD_COUNT
#define WRITE_VAL(LOC, ITER, VAL) buffer[tid + (LOC)*THREAD_COUNT + (ITER)*THREAD_COUNT*LOC_COUNT] = 0xA0000000 | (tid << 24) | (ITER << 16) | (LOC << 8) | VAL
#define WRITE(LOC, ITER) WRITE_VAL(LOC, ITER, WaveGetActiveMask())

//TEST_INPUT:cbuffer(data=[0 1]):name C
cbuffer C
{
	int alwaysFalse;
	int alwaysTrue;
}

void test(uint tid)
{
	for(int ii = 0; ii < tid; ++ii)
	{
		WRITE(0, ii);
		if(tid & 1)
		{
			if(tid & 2)
			{
				// Note: because of the two `if(alwaysFalse) break;` branches
				// here, the `WRITE(3,ii)` at the end of the loop body no
				// longer post-dominates the entry to the `if`, so that
				// implemenations that use immediate post-dominator
				// reconvergence will not properly reconvergence lanes
				// one and two at that point.
				//
				// The Slang synthesis pass for the active mask produces
				// the expected/desired result even in the presence of
				// these additional control-flow edges.
			#ifndef HACK
				if(alwaysFalse) break;
			#endif

				WRITE(1, ii);
				continue;
			}
			else
			{
			#ifndef HACK
				if(alwaysFalse) break;
			#endif

				WRITE(2, ii);
			}
		}
		WRITE(3, ii);
	}
	WRITE(4, 0);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}