//TEST:REFLECTION:-target glsl -profile ps_4_0 -entry main -fvk-t-shift 10 all -fvk-s-shift 100 all -fvk-b-shift 0 all -fvk-u-shift 1000 all

struct Data
{
    float a;
    int b;
};

Texture2D 		t;
SamplerState 	s;
ConstantBuffer<Data> c;
Texture2D t2;

RWStructuredBuffer<Data> u;
RWStructuredBuffer<int> u2;

float4 main() : SV_TARGET
{
	return float4(1, 1, 1, 0);
}