// entry-point-uniform-params.slang

// Confirm that `uniform` parameters on
// entry points are allowed, and work as expected.

//DISABLE_TEST:CPU_REFLECTION: -profile cs_5_0 -entry computeMain -target cpp
//TEST(compute):COMPARE_COMPUTE: -dx12 -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE: -dx11 -shaderobj
//TEST(compute):COMPARE_COMPUTE: -cuda -shaderobj
//TEST(compute):COMPARE_COMPUTE: -cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE: -metal -shaderobj

struct Signs
{
	int a;
}

struct Stuff
{
	int b;
}

struct Things
{
	int c;
}

// A shader parameter at global scope should be assigned
// a register/binding before any related to the entry point.

//TEST_INPUT:cbuffer(data=[1 0 0 0]):name=signs
ConstantBuffer<Signs> signs;

[numthreads(4, 1, 1)]
void computeMain(
//TEST_INPUT:uniform(data=[2]):name=stuff
	uniform Stuff  stuff,
//TEST_INPUT:uniform(data=[3]):name=things
	uniform Things things,

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
	uniform RWStructuredBuffer<int> outputBuffer,

	int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;

	int val = 0;
	val = val*16 + signs.a;
	val = val*16 + stuff.b;
	val = val*16 + things.c;
	val = val*16 + tid;

	outputBuffer[tid] = val;
}
