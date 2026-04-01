// shared-modifier.hlsl
//TEST:REFLECTION:-profile ps_5_0 -target hlsl -no-codegen

// Confirm that we expose the `shared` modifier in reflection data.

Texture2D t;
shared SamplerState s;

float4 main(float2 uv : UV) : SV_Target
{
	return t.Sample(s, uv);
}