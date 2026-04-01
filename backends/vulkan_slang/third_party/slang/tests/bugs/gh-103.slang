//TEST:COMPARE_HLSL: -profile ps_4_0 -entry main

// Ensure that matrix-times-scalar works

#ifdef __SLANG__
#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD
#else
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define END_CBUFFER(NAME, REG) ; cbuffer NAME : REG { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD

#define C C_0
#define a a_0
#define b b_0
#endif

float4x4 doIt(float4x4 a, float b)
{
    return a * b;
}

BEGIN_CBUFFER(C)
{
    float4x4 a;
    float b;    
}
END_CBUFFER(C,register(b0))

float4 main() : SV_TARGET
{
    return doIt(
        CBUFFER_REF(C,a),
        CBUFFER_REF(C,b))[0];
}
