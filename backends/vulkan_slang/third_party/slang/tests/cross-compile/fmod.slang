// fmod.slang

// Ensure that HLSL `fmod` works and produces
// expected output on Vulkan/GLSL.

//TEST(compute):COMPARE_COMPUTE:-dx11 -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -compute -shaderobj -emit-spirv-via-glsl
//TEST(compute):COMPARE_COMPUTE:-vk -compute -shaderobj -emit-spirv-directly

//TEST_INPUT:cbuffer(data=[4 0 0 0]):name=C
cbuffer C
{
	int y;
}

int test(int x)
{
	return int(fmod(float(x), float(y)));
}

//TEST_INPUT:ubuffer(data=[-7 -2 2 7], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    outputBuffer[tid] = test(outputBuffer[tid]);
}
