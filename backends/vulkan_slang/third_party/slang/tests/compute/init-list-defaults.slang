// init-list-defaults.slang
//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

// Confirm that initializer lists correctly default-initialize elements past those specified.

struct Test
{
	int4 a;
	int b[4];
}

int test(int inVal)
{
	Test myArray[4] = {
		{ int4(1), { 2, 3} },
		{ {4, 5, 6, }, { 7, } },
	};

	return myArray[0].b[inVal]
		+ myArray[1].a[inVal]*16
		+ myArray[inVal].a.x*256
		+ (inVal+1)*4096;
}

//TEST_INPUT:ubuffer(data=[9 9 9 9], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	int inVal = int(tid);
	int outVal = test(inVal);

	outputBuffer[tid] = outVal;
}