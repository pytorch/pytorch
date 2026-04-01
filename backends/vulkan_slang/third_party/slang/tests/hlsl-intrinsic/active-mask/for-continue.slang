// for-continue.slang

// Test case of a `for` loop that
// has multiple paths to continue
// the loop (both the ordinary one
// and an explicit `continue`)

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute 
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_0 -xslang -DHACK
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -xslang -DHACK
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -render-features cuda_sm_7_0

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

// In order to make a test of `continue` behavior
// adversarial, we need to observe the value of
// the active mask during the code that executes
// on a `continue`.
//
// We therefore define a subroutine that
// will perform the increment action for an
// ordinary counted loop, allowing us to
// observe the value of the active mask inside
// the function.
//
void inc(uint tid, in out int ii)
{
	// NOTE: For our current HLSL and GLSL output
	// strategies, we end up duplicating the
	// "continue clause" of a `for` loop into
	// each of the sites in the code where control
	// flow continues the loop. Unsurprisingly,
	// those copies mean that the active mask
	// seen on those platforms is not the expected
	// one.
	//
	// We will therefore write out the expected
	// value directly instead of using `WaveGetActiveMask()`
	// on those targets.
	//
#ifdef HACK
	WRITE_VAL(3, ii, (0xE << ii) & 0xF);
#else
	WRITE(3, ii);
#endif
	ii++;
}

void test(uint tid)
{
	for(int ii = 0; ii < tid; inc(tid, ii))
	{
		WRITE(0, ii);
		if(tid & 1)
		{
			WRITE(1, ii);
			continue;
		}
		WRITE(2, ii);
	}
	WRITE(4, 0);
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	test(dispatchThreadID.x);
}