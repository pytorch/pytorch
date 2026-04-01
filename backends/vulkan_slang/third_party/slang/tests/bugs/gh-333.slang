//TEST:COMPARE_HLSL: -profile ps_5_0 -entry main

// Ensure declaration order in output is correct

#ifdef __SLANG__
#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD
#else
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define END_CBUFFER(NAME, REG) ; cbuffer NAME : REG { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD

#define A A_0
#define x x_0
#define B B_0
#define y y_0
#define C C_0
#define a a_0
#define b b_0
#endif

struct A
{
	float x;
};

struct B
{
	float y;
	Texture2D t;
};

BEGIN_CBUFFER(C)
{
	A a;
	B b;
}
END_CBUFFER(C,register(b0))

float4 main() : SV_TARGET
{
	return CBUFFER_REF(C,a).x;
}
