// static-const-array.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name outputBuffer
RWStructuredBuffer<int> outputBuffer;

static const int kArray[] = { 16, 1, 32, 2 };

int test(int val)
{
	return kArray[val];
}

[numthreads(4, 1, 1)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
	int inVal = tid.x;
	int outVal = test(inVal);
	outputBuffer[inVal] = outVal;
}
