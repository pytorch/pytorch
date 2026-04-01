// func-resource-result-simple.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl

// Test that a function that returns a resource type can be
// compiled for targets that don't natively support resource
// return values.

//TEST_INPUT:ubuffer(data=[0x11 0x22 0x33 0x44], stride=4):name=inputBuffer
RWStructuredBuffer<int> inputBuffer;

RWStructuredBuffer<int> getInputBuffer() { return inputBuffer; }

int test(int val)
{
	return getInputBuffer()[val];
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
