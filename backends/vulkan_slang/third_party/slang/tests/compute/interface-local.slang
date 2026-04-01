// interface-local.slang

// Test basic use of an interface as an existential type
// for a local variable, instead of just as a constraint
// on a generic type parameter.
//
// Because the existential is created and then used inside
// the same function, we can eliminate it via simple
// local optimizations.

// on a generic type parameter.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj


interface IHelper
{
	int getVal();
}

struct HelperImpl : IHelper
{
	int storedVal;

	int getVal() { return storedVal; }
}

int test(int val)
{
    HelperImpl helperImpl = { val };

    IHelper existentialHelper = helperImpl;

    return existentialHelper.getVal();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inputVal = int(tid);
	int outputVal = test(inputVal);
	gOutputBuffer[tid] = outputVal;
}