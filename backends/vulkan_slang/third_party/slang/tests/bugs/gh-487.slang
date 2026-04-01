// gh-487.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

// This test is to confirm that we can apply builtin functions taht expect
// a floating-point argument to an integer, with the compiler filling
// in the implicit conversion. This is made tricky by the fact
// that a builtin line `sqrt` is actually a constrained generic,
// `sqrt<T:BuiltinFloatingPointType>` so that inference currently
// fails to deduce `T=float` when presented with an `int` argument.

int test(int val)
{
	int squared = val * val;
	float result = sqrt(squared);
	return int(result);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name gBuffer
RWStructuredBuffer<int> gBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	int inVal = int(tid);
	int outVal = test(inVal);

	gBuffer[tid] = outVal;
}