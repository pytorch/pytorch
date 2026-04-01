// mutating-methods.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -xslang -serial-ir -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -xslang -serial-ir -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -xslang -serial-ir -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -xslang -serial-ir -shaderobj

interface IAccumulator
{
	[mutating] void accumulate(int v);
}

struct Accumulator : IAccumulator
{
    int state;

	[mutating] void accumulate(int v)
	{
		state += v;
	}

	[mutating] void accumulateMore(int v)
	{
		this.state += v;
	}
}

void doStuff<A : IAccumulator>(inout A a)
{
    a.accumulate(16);
}

int test(int x)
{
    Accumulator a;
    a.state = 0;

    a.accumulate(x);
    a.accumulateMore(x);
    doStuff(a);

    return a.state;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
	outputBuffer[tid] = test(tid);
}
