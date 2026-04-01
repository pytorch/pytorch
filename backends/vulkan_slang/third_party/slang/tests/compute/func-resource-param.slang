// func-resource-param.slang

// Test that a function with a resource parameter that
// requires non-trivial legalization can be compiled
// to work on GLSL-based targets.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-dx12 -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

//NO_TEST:SIMPLE:-target glsl -entry computeMain -stage compute -validate-ir -dump-ir

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[0 16 32 48], stride=4):name=inputBuffer
RWStructuredBuffer<int> inputBuffer;

int helper(RWStructuredBuffer<int> buffer, int index)
{
	return buffer[index];
}

int test(int val)
{
    return helper(inputBuffer, val) + val;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int inVal = (int) dispatchThreadID.x;
    int outVal = test(inVal);
    outputBuffer[dispatchThreadID.x] = outVal;
}