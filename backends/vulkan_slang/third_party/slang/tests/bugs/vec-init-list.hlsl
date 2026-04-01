//TEST:COMPARE_HLSL: -profile vs_5_0

// Check handling of initializer list for vector

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
#define SV_Position SV_POSITION

#endif

BEGIN_CBUFFER(C)
{
    float4 a;
}
END_CBUFFER(C, register(b0))

float w0(float x) { return x; }
float w1(float x) { return x; }
float w2(float x) { return x; }
float w3(float x) { return x; }

float4 main() : SV_Position
{
    float4 wx = {
        w0(CBUFFER_REF(C,a).x),
        w1(CBUFFER_REF(C,a).x),
        w2(CBUFFER_REF(C,a).x),
        w3(CBUFFER_REF(C,a).x), };
    return wx;
}
