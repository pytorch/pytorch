//TEST:REFLECTION:-target glsl -profile ps_4_0 -entry main -fvk-t-shift 5 all -fvk-t-shift 7 2  -fvk-s-shift -3 0 -fvk-b-shift 1 2

Sampler2D<float> t0 : register(t2);
Sampler2D<float4> t1 : register(t7, space2);

float4 main() : SV_TARGET
{
	return float4(1, 1, 1, 0);
}