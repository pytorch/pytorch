//TEST:REFLECTION:-profile ps_4_0 -target hlsl -no-codegen

// Confirm that we handle global generic parameters



interface IBase
{};

type_param TParam : IBase;
type_param TParam2 : IBase;

struct S
{
	TParam2 p;
};

ParameterBlock<S> arg;
ParameterBlock<TParam> arg1;

Texture2D t;
SamplerState s;

cbuffer CB
{
	float4 u;
	float4 v;
	float4 w;
}


float4 main() : SV_Target
{
	return u + v + w + t.Sample(s, u.xy);
}