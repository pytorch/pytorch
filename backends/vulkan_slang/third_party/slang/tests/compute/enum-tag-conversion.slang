// enum-tag-conversion.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

// Confirm that a value of `enum` type can have an initializer
// that includes basic operations like type conversion.

enum RoseColors
{
    Red = 16u,
}

int test(int val)
{
	return val + int(RoseColors.Red);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    int val = int(tid);
    val = test(val);

    outputBuffer[tid] = val;
}