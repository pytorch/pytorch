//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

// Test that we correctly support both `out`
// and `inout` function parameters.

void testOut(int x, out int y)
{
	y = x;
}

void testInOut(int x, in out int y)
{
	y = y + x;
}

void testInout(int x, inout int y)
{
	y = y + x;	
}

int test(int inVal)
{
	int x0 = inVal;
	int x1;

	testOut(x0, x1);

	int x2 = x0;
	testInOut(x1, x2);

	int x3 = x0;
	testInout(x2, x3);

	return x3;
}

//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inVal = outputBuffer[tid];
	int outVal = test(inVal);
	outputBuffer[tid] = outVal;
}