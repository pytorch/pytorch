//DISABLED_TEST(compute):COMPARE_COMPUTE:

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out

//TEST_INPUT:cbuffer(data=[256]):
//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):

//TEST_INPUT:cbuffer(data=[65536]):

//TEST_INPUT:cbuffer(data=[4096]):
//TEST_INPUT:ubuffer(data=[16 32 48 64], stride=4):

// Test that we can declare a `ParameterBlock<...>` type as a shader
// parameter (potentially nested inside a `cbuffer`) and use it in
// shader code processed by the "rewriter"

import rewriter_parameter_block;

RWStructuredBuffer<int> outputBuffer : register(u0);

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inVal = tid;

	int outVal = test(gA, inVal) + test(gB, inVal) + extra;

	outputBuffer[tid] = outVal;
}