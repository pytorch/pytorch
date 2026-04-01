// entry-point-uniform-params-implicit.slang

// Test that slang can treat a compute shader parameter as `uniform` without explicit `uniform` keyword.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj  -xslang -Wno-38040
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -xslang -Wno-38040
//TEST:SIMPLE(filecheck=WARNING): -target spirv

struct Data
{
	int a;
	int b;
}

int test(int val, int a, int b)
{
	return a*(val+1) + b*(val+2);
}

[numthreads(4, 1, 1)]
[shader("compute")]
void computeMain(

//TEST_INPUT:uniform(data=[256 1]):name=d
// WARNING: ([[# @LINE+1]]): warning 38040
	Data d,

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
	uniform RWStructuredBuffer<int> outputBuffer,

	int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal, d.a, d.b);
    outputBuffer[tid] = outVal;

    // CHECK: 102
    // CHECK: 203
    // CHECK: 304
    // CHECK: 405
}
