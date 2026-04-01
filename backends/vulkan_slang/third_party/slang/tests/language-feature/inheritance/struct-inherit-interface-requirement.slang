// struct-inherit-interface-requirement.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj

// Test that a `struct` type can use an inherited
// member to satisfy an interface requirement.

interface ITweak
{
	int tweak(int val);
	int twiddle(int val);
}

// Note: `Base` intentionally doesn't inherit from `ITweak`,
// but it *does* provide a method that could satisfy one
// of the interface requirements.
//
struct Base
{
	int a;

	int tweak(int val) { return val ^ a; }
}

struct Derived : Base, ITweak
{
	// Note: it is important for this type to have an additional
	// field beyond the one in `Base`, because it ensures that
	// the two types `Base` and `Derived` aren't structurally
	// equivalent when compiled through HLSL (which silently allows
	// certain type mismatches so long as there is a memberwise
	// structural match).
	int b;

	int twiddle(int val)
	{
		return val + b;
	}
}

int tweakAndTwiddle<T : ITweak>(T tweaker, int val)
{
	int tmp = val;
	tmp = tweaker.tweak(tmp);
	tmp = tweaker.twiddle(tmp);
	return tmp;
}


int test(int val)
{
	Derived d;
	d.a = 0xFF;
	d.b = 1;

	return tweakAndTwiddle(d, val);
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
