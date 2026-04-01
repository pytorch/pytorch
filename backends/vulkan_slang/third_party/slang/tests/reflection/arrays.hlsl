//TEST:REFLECTION:-profile ps_4_0 -target hlsl -no-codegen

// Confirm that we can generate reflection info for arrays
//
// Note: just working with fixed-size arrays for now.
// Unbounded arrays may require more work.

cbuffer MyConstantBuffer
{
	float x;

	float a[10];

	float y;
}

Texture2D tx;
Texture2D ta[16];
Texture2D ty;
SamplerState sx;
SamplerState sa[4];
SamplerState sy;

float4 main() : SV_Target
{
	return 0.0;
}