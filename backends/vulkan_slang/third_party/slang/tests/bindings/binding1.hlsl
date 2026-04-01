//TEST:COMPARE_HLSL: -profile ps_4_0 -entry main

// We want to make sure that the registers Slang generates
// are used, even if there are "dead" parameter earlier in the program.
//
// In this case, we declare two each of textures, samplers, and constant
// buffers, and then only use the second one.
// Left to its own devices, the HLSL compiler would usually shift the
// object that was used up to binding slot zero, and eliminate the one
// that wasn't used.
// We expect Slang to generate explicit annotations that stop this from
// happening.

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

#define tB tB_0
#define sB sB_0

#define C1 C1_0
#define c1 c1_0

#endif

float4 use(float4 val) { return val; };
float4 use(Texture2D t, SamplerState s) { return t.Sample(s, 0.0); }

Texture2D 		tA R(: register(t0));
Texture2D 		tB R(: register(t1));
SamplerState 	sA R(: register(s0));
SamplerState 	sB R(: register(s1));

BEGIN_CBUFFER(C0)
{
	float c0;
}
END_CBUFFER(C0, register(b0))

BEGIN_CBUFFER(C1)
{
	float c1;
}
END_CBUFFER(C1, register(b1))

float4 main() : SV_TARGET
{
	return use(tB,sB) + use(CBUFFER_REF(C1,c1));
}