// comma-operator.slang

//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that the "comma operator" behaves as expected

// We will also include a cross-compilation test just to
// confirm that the generated SPIR-V (and by extension DXIL/DXBC)
// doesn't include a subroutine defintion/call for `operator,`

//TEST:CROSS_COMPILE:-target spirv-assembly -entry computeMain -stage compute

int test(int inVal)
{
	int a = inVal;

	// We expect the left operand to `operator,` to be
	// evaluated before the right operand, and for the
	// result to be the value of the right operand
	//
	return a*=2, a+1;
}

//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inVal = outputBuffer[tid];
	int outVal = test(inVal);
	outputBuffer[tid] = outVal;
}