//TEST:COMPARE_HLSL: -profile ps_5_1 -entry main

// Let's first confirm that Slang can reproduce what the
// HLSL compiler would already do in the simple case (when
// all shader parameters are actually used).

float4 use(Texture2D t, SamplerState samp) { return t.Sample(samp, 0.0); }

#ifdef __SLANG__

struct Test
{
	Texture2D a;
	Texture2D b;
};

Test test[2];
SamplerState s;

float4 main() : SV_Target
{
	return use(test[0].a,s)
		 + use(test[0].b,s)
		 + use(test[1].a,s)
		 + use(test[1].b,s);
}

#else

#define a test_a_0
#define b test_b_0
#define s s_0

Texture2D a[2];
Texture2D b[2];
SamplerState s;

float4 main() : SV_TARGET
{
	return use(a[0],s)
		 + use(b[0],s)
		 + use(a[1],s)
		 + use(b[1],s);
}

#endif
