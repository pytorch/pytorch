//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=HLSL): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none
//TEST:SIMPLE(filecheck=GLSL): -target glsl -profile glsl_450 -stage compute -entry computeMain -line-directive-mode none

// Metal does not support custom data layout.
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT:set Constants.v0={1.0,2.0,3.0,4.0}
//TEST_INPUT:set Constants.v1={5.0,6.0,7.0}
//TEST_INPUT:set Constants.v2=8.0

cbuffer Constants
{
    float4 v0 : packoffset(c0);
    float3 v1 : packoffset(c1);
    float v2 : packoffset(c1.w);
};
// HLSL: cbuffer
// HLSL: {
// HLSL: {{.*}} : packoffset(c0);
// HLSL: {{.*}} : packoffset(c1);
// HLSL: {{.*}} : packoffset(c1.w);
// HLSL: }
// GLSL: layout(offset = 0)
// GLSL: layout(offset = 16)
// GLSL: layout(offset = 28)

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadID.x] = v2;
}
