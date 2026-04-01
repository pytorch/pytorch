//TEST:REFLECTION:-target glsl -profile ps_4_0 -entry main -fvk-bind-globals 5 9

uniform int a;
uniform float b;

Texture2D t;
SamplerState sampler;

float4 main() : SV_TARGET
{
	return t.SampleLevel(sampler, float2(a,b), 0) + float4(a, b, 1, 0);
}