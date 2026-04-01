//TEST(smoke):REFLECTION:-profile glsl_fragment -target glsl -no-codegen

// Confirm that when targetting GLSL via cross compilation,
// we use the Vulkan layout rules instead of HLSL ones
//
// Specifically, we expect all of our top-level parameters
// to get distinct registers in the same descriptor set,
// rather than all getting register `0` because they
// occupy different names register spaces (as they would in D3D)

Texture2D t;

SamplerState s;

cbuffer C
{
	float3 c;
}

float4 main() : SV_Target
{
	return 0.0;
}