// struct-inheritance.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj

// Test that we can define a `struct` type
// that inherits from another `struct`.

struct Base
{
	int a;

	int tweakBase(int val) { return val ^ a; }
}

struct Derived : Base
{
	int b;

	int tweakDerived(int val) { return tweakBase(val) + b; }
}

int tweak(Base b, int v)
{
	return b.tweakBase(v);
}

//TEST_INPUT:cbuffer(data=[1 2]):name=C
cbuffer C
{
	int x;
	int y;
}

int test(int val)
{
	Derived d;// = { x, y };
	d.a = x;
	d.b = y;

	int result = 0;
	result = result*16 + d.a;
	result = result*16 + d.tweakBase(val);
	result = result*16 + tweak(d, val);
	result = result*16 + d.tweakDerived(val);
	return result;
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
