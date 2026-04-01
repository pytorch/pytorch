// user-defined-initializer.slang

// Confirm that user-defined initializer/constructor
// methods in a type work as expected.

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

struct Pair
{
	int head;
	int tail;

	__init(int h, int t)
	{
		head = h;
		tail = t;
	}

	int getHead() { return head; }
	int getTail() { return tail; }
}

int test(int value)
{
	Pair p = Pair(value, value+1);
	return p.getHead()*16 + p.getTail();
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