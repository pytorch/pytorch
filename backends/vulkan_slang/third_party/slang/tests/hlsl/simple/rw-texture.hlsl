// rw-texture.hlsl

//TEST:COMPARE_HLSL: -profile ps_5_0 -entry main

// Ensure that we implement the `Load` operations on
// `RWTexture*` types with the correct signature.

#ifdef __SLANG__
#define R(X) /**/
#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD
#else
#define R(X) : register(X)
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define END_CBUFFER(NAME, REG) ; cbuffer NAME : REG { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD
#define C C_0
#define SV_Target SV_TARGET
#define u2 u2_0
#define u3 u3_0
#define t2 t2_0
#define t2a t2a_0
#define t3 t3_0
#endif


BEGIN_CBUFFER(C)
{
    uint2 u2;
    uint3 u3;
}
END_CBUFFER(C, register(b0))

RWTexture2D<float4>         t2  R(u1);
RWTexture2DArray<float4>    t2a R(u2);
RWTexture3D<float4>         t3  R(u3);

float4 main() : SV_Target
{
    return t2.Load(int2(CBUFFER_REF(C,u2)))
        + t2a.Load(int3(CBUFFER_REF(C,u3)))
        +  t3.Load(int3(CBUFFER_REF(C,u3)));
}
