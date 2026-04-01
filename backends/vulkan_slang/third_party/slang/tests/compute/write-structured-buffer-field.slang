//TEST(compute):COMPARE_COMPUTE: -shaderobj


struct S
{
	int4 a;
	int4 b;
};

//TEST_INPUT:ubuffer(data=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32], stride=32):out,name=outputBuffer
RWStructuredBuffer<S> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	outputBuffer[tid].b.y = outputBuffer[0].a.x;
}