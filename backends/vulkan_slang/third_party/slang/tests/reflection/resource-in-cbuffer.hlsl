//TEST(smoke):REFLECTION:-profile ps_4_0 -target hlsl -no-codegen

// Confirm that we can generate reflection
// information for resources nested inside
// a cbuffer:

cbuffer MyConstantBuffer
{
	float3 v;

	Texture2D myTexture;

	float  c;

	SamplerState mySampler;
}

float4 main() : SV_Target
{
	return 0.0;
}