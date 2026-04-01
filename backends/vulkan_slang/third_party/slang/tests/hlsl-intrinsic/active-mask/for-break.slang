// for-break.slang

// Test active mask synthesis for a `for` loop that
// has no ordinary exit condition and can thus
// only be exited via a (single) `break`.

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
#define WRITE(LOC, ITER) buffer[tid + (LOC)*THREAD_COUNT + (ITER)*THREAD_COUNT*LOC_COUNT] = 0xA0000000 | (tid << 24) | (ITER << 16) | (LOC << 8) | WaveGetActiveMask()

//TEST_INPUT:cbuffer(data=[0 1]):name C
cbuffer C
{
	int alwaysFalse;
	int alwaysTrue;
}

void test(int tid)
{
	int ii = 0;
	for(;;)
	{
		WRITE(0, ii);
		if(ii >= tid)
		{
			WRITE(1, ii);

			// Note: This flag has been included to force
			// D3D and Vulkan implementations to provide
			// the expected/desired behavior for the active
			// mask in the preceding code.
			//
			// It seems that without making the `break`
			// conditional, some implementation will see that
			// this block post-dominates the entire loop,
			// and thus decide that the code is semantically
			// "outside" the loop, despite the fact that
			// it is clearly lexically *inside* the loop.
			//
			// Making the `break` conditional introduces
			// a new control-flow edge that changes the
			// post-dominator relationship and thus makes
			// such implementations see this code as being
			// "inside" the loop again.
			//
		#ifdef HACK
			if(alwaysTrue)
		#endif
			break;
		}
		WRITE(2, ii);
		ii++;
	}
	WRITE(3, ITER_COUNT);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}