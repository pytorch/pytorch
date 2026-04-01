//TEST:COMPARE_HLSL: -profile ps_4_0 -entry main

// Confirm that resources inside constant buffers get correct locations,
// including the case where there are *multiple* constant buffers
// with resources.

#ifdef __SLANG__
#define R(X) /**/
#define BEGIN_CBUFFER(NAME) cbuffer NAME {
#define MID_CBUFFER(NAME) /**/
#define END_CBUFFER(NAME, REG) /**/ }
#define CBUFFER_REF(NAME, FIELD) FIELD
#else
#define R(X) X
#define BEGIN_CBUFFER(NAME) struct SLANG_ParameterGroup_##NAME {
#define MID_CBUFFER(NAME) };
#define END_CBUFFER(NAME, REG) cbuffer NAME : REG { SLANG_ParameterGroup_##NAME NAME; }
#define CBUFFER_REF(NAME, FIELD) NAME.FIELD

#define CA CA_0
#define caa caa_0
#define cab cab_0
#define cac cac_0
#define cad cad_0
#define cae cae_0
#define ta 	CA_ta_0
#define sa 	CA_sa_0

#define CB CB_0
#define cba cba_0
#define cbb cbb_0
#define cbc cbc_0
#define cbd cbd_0
#define cbe cbe_0
#define tbx	CB_tbx_0
#define tby	CB_tby_0
#define sb 	CB_sb_0

#define CC CC_0
#define cca cca_0
#define ccb ccb_0
#define ccc ccc_0
#define ccd ccd_0
#define cce cce_0
#define tc 	CC_tc_0
#define scx	CC_scx_0
#define scy	CC_scy_0

#endif

float4 use(float  val) { return val; };
float4 use(float2 val) { return float4(val,0.0,0.0); };
float4 use(float3 val) { return float4(val,0.0); };
float4 use(float4 val) { return val; };
float4 use(Texture2D t, SamplerState s) { return t.Sample(s, 0.0); }

BEGIN_CBUFFER(CA)

	float4 caa;
	float3 cab;
	float  cac;
	float2 cad;
	float2 cae;

MID_CBUFFER(CA)

	Texture2D ta R(: register(t0));
	SamplerState sa R(: register(s0));

END_CBUFFER(CA, register(b0))

BEGIN_CBUFFER(CB)

	float4 cba;
	float3 cbb;
	float  cbc;
	float2 cbd;
	float2 cbe;

MID_CBUFFER(CB)

	Texture2D tbx R(: register(t1));
	Texture2D tby R(: register(t2));
	SamplerState sb R(: register(s1));

END_CBUFFER(CB, register(b1))

BEGIN_CBUFFER(CC)

	float4 cca;
	float3 ccb;
	float  ccc;
	float2 ccd;
	float2 cce;

MID_CBUFFER(CC)

	Texture2D tc R(: register(t3));
	SamplerState scx R(: register(s2));
	SamplerState scy R(: register(s3));

END_CBUFFER(CC, register(b2))

float4 main() : SV_TARGET
{
	// Go ahead and use everything in this case:
	return use(ta,  sa)
		+  use(tbx, sb)
		+  use(tby, scx)
		+  use(tc,  scy)
		+  use(CBUFFER_REF(CA, cae))
		+  use(CBUFFER_REF(CB, cbe))
		+  use(CBUFFER_REF(CC, cce))
		;
}