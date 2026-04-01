// global-generic-value-param.slang

//DISABLED_TEST(compute):COMPARE_COMPUTE: -shaderobj

// This is a basic test of support for global generic
// value parameters: explicit named parameters at global
// scope that can be used to generate specialized kernel
// code based on different values.

// We start by declaring a global generic value parameter:
//
// Note: only `int` parameters are expected to work for now.
// Note: the default `= 0` intializer isn't used right now.
//
__generic_value_param kOffset : uint = 0;

// For the test framework, we also need to specify what
// value we want to specialize to.
//
// Note: this value (7) will be fed in to the compiler API
// as a specialization argument, and will not be visible
// to the compiler when it initially compiles the code
// to IR.
//
//TEST_INPUT: globalSpecializationArg 7

// Next we will declare a buffer of data just so that we
// can index into something and make the shader logic a
// bit less trivial.
//
RWStructuredBuffer<uint> vals;
//TEST_INPUT: ubuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], stride=4):name vals

// The core test function will use the `kOffset` value
// we declared above along with the input value (the
// thread ID) to index into our buffer of values and
// compute a result. All of the math here is just to
// make the result easy to validate by eye.
//
uint test(uint value)
{
	return value * 16 + vals[(value + kOffset) & 0xF];
}

// And finally we have the boilerplate cruft that almost
// all of our compute tests use.

//TEST_INPUT: ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(16, 1, 1)]
void computeMain(
	uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	uint inVal = tid;
	uint outVal = test(inVal);
	outputBuffer[tid] = outVal;
}