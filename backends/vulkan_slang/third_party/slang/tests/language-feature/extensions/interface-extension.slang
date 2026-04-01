// interface-extension.slang

// Test that an `extension` applied to an interface type works as users expect

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj

interface ICounter
{
	[mutating] void add(int value);
}

struct MyCounter : ICounter
{
	int _state = 0;

	[mutating] void add(int value) { _state += value; }
}

__generic<T : ICounter>
extension T
{
	[mutating] void increment()
	{
	    this.add(1);
	}
}

void helper<T : ICounter>(in out T counter)
{
	counter.increment();
}

int test(int value)
{
	MyCounter counter = { value };
	counter.increment();
	helper(counter);
	return counter._state;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
