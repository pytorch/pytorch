// global-uniform-params.slang

//DISABLED_TEST(compute):COMPARE_COMPUTE:

// Test that code can use uniform parameters
// of "ordinary" type declared at the global scope

//TEST_INPUT:cbuffer(data=[256 1]):name=$Globals
uniform int a;
uniform int b;

int test(int val)
{
	return a*(val+1) + b*(val+2);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
[shader("compute")]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
