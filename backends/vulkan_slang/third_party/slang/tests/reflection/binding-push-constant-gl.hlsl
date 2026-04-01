//TEST:REFLECTION:-profile ps_4_0 -target spirv -no-codegen

// Confirm that we can generate reflection info for arrays
//
// Note: just working with fixed-size arrays for now.
// Unbounded arrays may require more work.

layout(binding=0, set = 1) cbuffer MyConstantBuffer
{
	float x;

	float a[10];

	float y;
}

struct MyPushConstantStruct
{
	float pushX;
	float pushY;
};

/* This style doesn't work
[[vk::push_constant]] MyPushConstantStruct pushConstantBuffer;
*/

// This style does work 
[[vk::push_constant]]ConstantBuffer<MyPushConstantStruct> pushConstantBuffer;
 
/* This style works too
layout(push_constant)  
cbuffer MyPushConstantBuffer
{
	float pushX;
	float pushY;
} 
*/
 
/* This works too
[[vk::push_constant]] cbuffer MyPushConstantBuffer
{
	float pushX;
	float pushY;
} 
*/

[[vk::binding(1,2)]] Texture2D tx;
[gl::binding(2,3)] Texture2D ta;
Texture2D ty;
SamplerState sx;
SamplerState sa[4];
SamplerState sy;

float4 main() : SV_Target
{
	return 0.0 + pushConstantBuffer.pushX;
}