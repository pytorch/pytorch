// struct-default-init.slang
//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj

struct Test
{
	int a = 0;
	int b = 1;
	int c = 0;
	int d = 1 + 1;
}

int test(int inVal)
{
	Test myArray[4] = {
		{ 3, 4, 5, 6 },
		{ 7, 8, 9, },
		{ 10, 11 },
		{ 12, }
	};

	Test t = myArray[inVal];
	return t.a * 4096
	     + t.b * 256
	     + t.c * 16
	     + t.d;
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
