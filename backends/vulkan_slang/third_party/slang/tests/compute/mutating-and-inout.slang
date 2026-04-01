// mutating-and-inout.slang

// Test that calling a `[mutating]` method on an `inout` function parameter works.

//TEST(compute):COMPARE_COMPUTE: -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct A
{
	int x;

	[mutating] void doThings(inout int y)
	{
		int tmp = x;
		x = y;
		y = tmp;
	}
}

int doThings(inout A a, int val)
{
	a.doThings(val);
	return val;
}

int test(int val)
{
	A a = { val };
	int b = val ^ 3;

	int c = doThings(a, b);

	int result = 0;
	result = result*16 + a.x;
	result = result*16 + b;
	result = result*16 + c;

	return result;
}

[numthreads(4, 1, 1)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
	int val = tid.x;
	val = test(val);
	outputBuffer[tid.x] = val;
}
