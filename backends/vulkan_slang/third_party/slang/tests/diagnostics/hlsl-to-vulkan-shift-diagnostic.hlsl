//DIAGNOSTIC_TEST:SIMPLE:-target glsl -profile ps_4_0 -entry main -fvk-t-shift 5 all -fvk-t-shift 7 2  -fvk-s-shift -3 0 -fvk-u-shift 1 2 -no-codegen

struct Data
{
    float a;
    int b;
};

Texture2D 		t : register(t0);
SamplerState 	s : register(s4);
ConstantBuffer<Data> c : register(b2);

Texture2D t2 : register(t0, space2);

RWStructuredBuffer<Data> u : register(u11);
RWStructuredBuffer<int> u2 : register(u3, space2);

float4 main() : SV_TARGET
{
	return float4(1, 1, 1, 0);
}
