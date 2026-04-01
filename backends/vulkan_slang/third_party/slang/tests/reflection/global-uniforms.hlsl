//TEST_IGNORE_FILE
//TEST:REFLECTION:-profile ps_4_0 -target hlsl

// Confirm that we handle uniforms at global scope


float4 u;

Texture2D t;
SamplerState s;

cbuffer CB
{
	float4 v;
}

float4 w;

float4 main() : SV_Target
{
	return u + v + w + t.Sample(s, u.xy);
}