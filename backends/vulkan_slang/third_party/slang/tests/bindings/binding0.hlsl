//TEST:COMPARE_HLSL: -profile ps_4_0 -entry main

// Let's first confirm that Slang can reproduce what the
// HLSL compiler would already do in the simple case (when
// all shader parameters are actually used).

#ifdef __SLANG__
#define R(X) /**/
#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD

#else
#define R(X) X
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define END_CBUFFER(NAME, REG) ; cbuffer NAME : REG { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD

#define C C_0
#define t t_0
#define s s_0
#define c c_0

#endif

float4 use(float4 val) { return val; };
float4 use(Texture2D tex, SamplerState samp) { return tex.Sample(samp, 0.0); }

Texture2D 		t R(: register(t0));
SamplerState 	s R(: register(s0));

BEGIN_CBUFFER(C)
{
	float c;
}
END_CBUFFER(C, register(b0))

float4 main() : SV_TARGET
{
	return use(t,s) + use(CBUFFER_REF(C,c));
}