//TEST:SIMPLE(filecheck=CHECK):-target hlsl -profile ps_5_1 -entry main -parameter-blocks-use-register-spaces

// Confirm that Slang `ParameterBlock<T>` generates
// parameter bindings like we expect.

// CHECK: Texture2D{{.*}} p_t_0 : register(t0);
// CHECK: Texture2D{{.*}} p_ta_0[{{.*}}] : register(t1);

// CHECK: Texture2D{{.*}} p1_t_0 : register(t0, space1);
// CHECK: SamplerState p1_s_0 : register(s0, space1);
float4 use(float4 val) { return val; };
float4 use(Texture2D t, SamplerState s) { return t.Sample(s, 0.0); }

#ifdef __SLANG__

struct S
{
	Texture2D 		t;
	Texture2D 		ta[4];
	SamplerState 	s;	
};

ParameterBlock<S> p;
ParameterBlock<S> p1;

float4 main(float v : V) : SV_Target
{
	return use(p.t, p.s)
		+ use(p.ta[int(v)], p.s)
		+ use(p1.t, p1.s)
		+ use(p1.ta[int(v)], p1.s);
}

#else

Texture2D p_t_0 : register(t0, space0);
Texture2D p_ta_0[4] : register(t1, space0);
SamplerState p_s_0 : register(s0, space0);

Texture2D p1_t_0 : register(t0, space1);
Texture2D p1_ta_0[4] : register(t1, space1);
SamplerState p1_s_0 : register(s0, space1);

float4 main(float v : V) : SV_TARGET
{
    int _S2 = int(v);
    return use(p_t_0, p_s_0)
		 + use(p_ta_0[_S2], p_s_0)
		 + use(p1_t_0, p1_s_0)
		 + use(p1_ta_0[_S2], p1_s_0);
}

#endif
