//TEST:REFLECTION:-profile ps_4_0  -target hlsl -no-codegen

// Confirm that basic reflection info can be output

float4 use(float4 val) { return val; };
float4 use(Texture2D t, SamplerState s) { return t.Sample(s, 0.0); }

Texture2D 		t;
SamplerState 	s;

cbuffer C
{
	float c;
}

float4 main() : SV_Target
{
	return use(t,s) + use(c);
}