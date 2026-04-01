//TEST:COMPARE_HLSL: -profile ps_4_0 -entry main

// We need to allow the user to add explicit bindings to their parameters,
// and we can't go and auto-assign anything to use the same locations.

#ifdef __SLANG__
#define R(X) /**/
#define BEGIN_CBUFFER(NAME) cbuffer NAME
#define BEGIN_CBUFFER_R(NAME, REG) cbuffer NAME : REG
#define END_CBUFFER(NAME, REG) /**/
#define CBUFFER_REF(NAME, FIELD) FIELD
#else
#define R(X) X
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME
#define BEGIN_CBUFFER_R(NAME, REG) BEGIN_CBUFFER(NAME)
#define END_CBUFFER(NAME, REG) ; cbuffer NAME : REG { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD

#define CA CA_0
#define ca ca_0

#define CB CB_0
#define cb cb_0

#define CC CC_0
#define cc cc_0

#define sa sa_0
#define sb sb_0
#define sc sc_0

#define ta ta_0
#define tb tb_0
#define tc tc_0

#endif

float4 use(float4 val) { return val; };
float4 use(Texture2D t, SamplerState s) { return t.Sample(s, 0.0); }

// We'll make three textures, but explicit assign the third one
// to the slot `t0`. We expect the others to shift further along
// to "make room".
Texture2D 		ta R(: register(t1));
Texture2D 		tb R(: register(t2));
Texture2D 		tc : register(t0);


// The explicit binding may "split" the range of register available
// for automatic placement. We use a "first-fit" approach to pack
// things in:
SamplerState 	sa R(: register(s0));
SamplerState 	sb R(: register(s2));
SamplerState 	sc : register(s1);

// It's also okay to use a register that *doesn't* conflict,
// and even to make things non-contiguous. Here we bind
// the third constnat buffer to register `b9`
//
BEGIN_CBUFFER(CA)
{
	float ca;
}
END_CBUFFER(CA, register(b0))

//
BEGIN_CBUFFER(CB)
{
	float cb;
}
END_CBUFFER(CB, register(b1))
//
BEGIN_CBUFFER_R(CC, register(b9))
{
	float cc;
}
END_CBUFFER(CC, register(b9))

float4 main() : SV_TARGET
{
	// Go ahead and use everything in this case:
	return use(ta, sa) + use(CBUFFER_REF(CA,ca))
		+  use(tb, sb) + use(CBUFFER_REF(CB,cb))
		+  use(tc, sc) + use(CBUFFER_REF(CC,cc));
}