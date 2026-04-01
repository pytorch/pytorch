//TEST:COMPARE_HLSL: -profile ps_5_0 -entry main

// Make sure we don't crash when desugaring resource in structs,
// when the user also declares multiple variables with a
// single declaration.

#ifdef __SLANG__

cbuffer C
{
    Texture2D t0, t1;
    SamplerState s;
    float2 uv;
};

float4 main() : SV_Target
{
    return t0.Sample(s, uv)
         + t1.Sample(s, uv);
}

#else

struct SLANG_ParameterGroup_C_0
{
    float2 uv_0;
};

cbuffer C_0 : register(b0)
{
    SLANG_ParameterGroup_C_0 C_0;
};

Texture2D C_t0_0 : register(t0);
Texture2D C_t1_0 : register(t1);
SamplerState C_s_0 : register(s0);

float4 main() : SV_TARGET
{
    return C_t0_0.Sample(C_s_0, C_0.uv_0)
         + C_t1_0.Sample(C_s_0, C_0.uv_0);
}

#endif


