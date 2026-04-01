//DISABLED_TEST(compute):COMPARE_COMPUTE:

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name=outputBuffer

//TEST_INPUT:cbuffer(data=[256]):name=C.gA.val
//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):name=C.gA.buf
//TEST_INPUT:cbuffer(data=[4096]):name=C.gB.val
//TEST_INPUT:ubuffer(data=[16 32 48 64], stride=4):name=C.gB.buf

//TODO_TEST_INPUT:object:name=C
//TODO_TEST_INPUT:object:name=C.gA
//TODO_TEST_INPUT:root_constants(data=[256]):name=C.gA.val
//TODO_TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):name=C.gA.buf
//TODO_TEST_INPUT:root_constants(data=[4096]):name=C.gB.val
//TODO_TEST_INPUT:ubuffer(data=[16 32 48 64], stride=4):name=C.gB.buf

// Test that we can declare a `ParameterBlock<...>` type as a shader
// parameter (potentially nested inside a `cbuffer`) and use it in
// shader code processed by the "rewriter"

import rewriter_parameter_block_complex;

RWStructuredBuffer<int> outputBuffer : register(u0);

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inVal = tid;

	int outVal = test(gA, inVal) + test(gB, inVal);

	outputBuffer[tid] = outVal;
}