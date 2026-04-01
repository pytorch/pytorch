//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):out,name outputBuffer

// Test that "operator comma" works as expected

static int a = 0;

int setA(int val)
{
	a = val;
	return 0;
}

int getA()
{
	return a;
}

int test(int inVal)
{
	int x = (setA(inVal), getA());
	return x * 16;
}

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inVal = outputBuffer[tid];
	int outVal = test(inVal);
	outputBuffer[tid] = outVal;
}