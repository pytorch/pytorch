// using-namespace.slang

// Test that `using` can bring declarations from a namespace into scope

//TEST(compute):COMPARE_COMPUTE: -shaderobj

namespace X
{
	struct A { int a; }
}

namespace Y
{
	// A `using` that brings one namespace into scope of declarations in another
	//
    using X;

	int b(A a) { return a.a; }
}

// A `using` that brings declarations in a namespace into the global scope
//
using Y;

int test(int value)
{
	// A `using` that brings declarations into a local scope
	// (and also uses the optional `namespace` placeholder keyword)
	//
	using namespace X;

	A a = { value };
	return b(a);
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

