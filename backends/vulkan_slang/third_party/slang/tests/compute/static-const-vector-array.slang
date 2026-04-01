// static-const-array.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -slang -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0  0 0 0 0], stride=4):out, name outputBuffer
RWStructuredBuffer<float> outputBuffer;

static const float3 kArray[8] =
{
    float3(-0.4706069, -0.4427112, - - + 0.6461146),
    float3(-0.9057375, +0.3003471, +0.9542373),
    float3(-0.3487388, +0.4037880, +0.5335386),
    float3(+0.1023042, +0.6439373, +0.6520134),
    float3(+0.5699277, +0.3513750, +0.6695386),
    float3(+0.2939128, -0.1131226, +0.3149309),
    float3(+0.7836658, -0.4208784, +0.8895339),
    float3(+0.1564120, -0.8198990, +0.8346850)
};

float test(int val)
{
	return kArray[val].x + kArray[val].y + kArray[val].z;
}

[numthreads(8, 1, 1)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
	int inVal = tid.x;
	float outVal = test(inVal);
	outputBuffer[inVal] = outVal;
}
