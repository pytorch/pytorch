// entry-point-uniform-params.slang
//TEST(compute):COMPARE_COMPUTE:-wgpu -shaderobj
//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Test that a shader can be written that
// only uses entry point `uniform` parameters,
// without any global parameters.

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
	uniform Data d,

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
	uniform RWStructuredBuffer<int> outputBuffer,

	int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal, d.a, d.b);
    outputBuffer[tid] = outVal;
}
