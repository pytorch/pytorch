//TEST(smoke,compute):COMPARE_COMPUTE: -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-cpu -shaderobj

// Test IR code generation for the `?:` "select" operator

int test(int input)
{
	return input > 1 ? -input : input;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadID.x] = test((int) dispatchThreadID.x);
}