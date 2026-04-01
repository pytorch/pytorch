//TEST:COMPARE_HLSL: -profile ps_5_0 -entry main
// Make sure we don't crash when desugaring resources
// in structs when a `cbuffer` only contains resources.

#ifdef __SLANG__

cbuffer C
{
	Texture2D t;
	SamplerState s;
};

float4 main(float2 uv: UV) : SV_Target
{
	return t.Sample(s, uv);
}

#else

Texture2D C_t_0 : register(t0);
SamplerState C_s_0 : register(s0);

float4 main(float2 uv: UV) : SV_TARGET
{
	return C_t_0.Sample(C_s_0, uv);
}

#endif


