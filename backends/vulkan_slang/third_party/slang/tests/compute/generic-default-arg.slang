//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

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

struct Impl2 : ITest
{
    int doThing(int x)
    {
        return x * 3;
    }
};

__generic<T : ITest = Impl1>
struct GenStruct
{
	T obj = T();
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
