//TEST:COMPARE_HLSL: -profile vs_5_0

// Confirm that type-cast expressions parse with
// the appropriate precedence.

#ifdef __SLANG__
#define R(X) /**/
#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD
#else
#define R(X) X
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define END_CBUFFER(NAME, REG) ; cbuffer NAME : register(REG) { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD

#define C C_0
#define a a_0
#define b b_0
#define SV_Position SV_POSITION
#endif

BEGIN_CBUFFER(C)
{
	float a;
	float b;
}
END_CBUFFER(C,b0)


float4 main() : SV_Position
{
	return (uint) CBUFFER_REF(C,a) / CBUFFER_REF(C,b);
}
