//TEST:COMPARE_HLSL: -profile ps_5_0

#ifdef __SLANG__

#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD

import split_nested_types;

#else

#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define END_CBUFFER(NAME, REG) ; cbuffer NAME : REG { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD

#define A A_0
#define x x_0

#define B B_0
#define y y_0

#define M M_0
#define a a_0
#define b b_0

#define C C_0
#define m m_0

struct A { int x; };

struct B { float y; };

struct CC { Texture2D t; SamplerState s; };

struct M
{
	A a;
	B b;
};

#endif

BEGIN_CBUFFER(C)
{
	M m;
}
END_CBUFFER(C,register(b0))

float4 main() : SV_TARGET
{
	return CBUFFER_REF(C,m).b.y;
}
