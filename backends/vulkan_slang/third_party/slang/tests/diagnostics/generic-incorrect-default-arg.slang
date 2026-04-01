//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none

// Check that user code can declare and use a generic
// `struct` type.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface ITest
{
	int doThing(int x);
};

struct Impl1 : ITest
{
	int doThing(int x)
	{
		return x * 2;
	}
};

struct Impl2
{
    int doSomethingElse(int x)
    {
        return x * 3;
    }
};

__generic<T : ITest = Impl2>
// CHECK: tests/diagnostics/generic-incorrect-default-arg.slang([[@LINE-1]]): error 38029: type argument 'Impl2' does not conform to the required interface 'ITest'
struct GenStruct
{
	T obj;
};

int test(GenStruct gs, int val)
{
	return gs.obj.doThing(val);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;

	int outVal = 0;

	GenStruct<Impl1> gs;
	outVal += test(gs, tid);

	outputBuffer[tid] = outVal;
}