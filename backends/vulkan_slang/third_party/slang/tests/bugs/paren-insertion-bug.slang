// paren-insertion-bug.slang

// Confirm that precedence is correctly handled
// for cast from scalar to vector.

//TEST(compute):COMPARE_COMPUTE: -shaderobj

int test(float a)
{
	// This line performs a cast from a scalar result to a vector
	float3 b = pow(a, 2.0);

	// If the computation of `b` above gets folded into this
	// line of code (and we expect it to) we need to correctly
	// parenthesize the generated cast so that the `.xyz` swizzle
	// applies to the result of the cast, rather than the input.
	//
	return int(float4(b.xyz * 2.0, 1.0).x);
}


//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	uint inVal = tid;
	uint outVal = test(inVal);

	outputBuffer[tid] = outVal;
}
