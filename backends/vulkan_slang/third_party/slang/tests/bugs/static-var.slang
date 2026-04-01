// gh-775-ext.slang
//TEST(compute):COMPARE_COMPUTE: -shaderobj

int test(int inVal)
{
    static int kVal = 16;
    return inVal + kVal;
}

//TEST_INPUT:ubuffer(data=[9 9 9 9], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	int inVal = int(tid);
	int outVal = test(inVal);

	outputBuffer[tid] = outVal;
}