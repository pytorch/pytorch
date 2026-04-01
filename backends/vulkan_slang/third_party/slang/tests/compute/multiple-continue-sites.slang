//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

// Test that a loop with multiple `continue` sites works.
//
// The current Slang codegen strategy for `continue` ends
// up duplicating the "continue clause" for a `for` loop
// at each `continue` site, so it will stress-test any
// code that assumes a given instruction/block only
// appears once in the region tree.
//

int test(int inVal)
{
	int ii = inVal;
	for(;!bool(ii & 0x20); ii += 0x10)
	{
		if(ii == 2)
		{
			continue;
		}

		ii += 0x100;
		// there is an implicit `continue` here
	}

	return ii;
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