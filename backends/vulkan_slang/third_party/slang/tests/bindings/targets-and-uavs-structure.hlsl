//TEST(smoke):COMPARE_HLSL: -profile ps_5_0 -entry main

// Handle the case where the fragment shader output is
// defined a structure, and the semantics are on the sub-fields

#ifdef __SLANG__
#define R(X) /**/
#else
#define R(X) X

#define Foo Foo_0
#define v v_0
#define fooBuffer fooBuffer_0

#endif

float4 use(float  val) { return val; };
float4 use(float2 val) { return float4(val,0.0,0.0); };
float4 use(float3 val) { return float4(val,0.0); };
float4 use(float4 val) { return val; };
float4 use(Texture2D t, SamplerState s) { return t.Sample(s, 0.0); }

struct Foo { float2 v; };

// This should be allocated a register *after* the render targets
RWStructuredBuffer<Foo> fooBuffer R(: register(u2));

struct Fragment
{
	float4 color : SV_Target0;
	float4 extra : SV_Target1;
	
};

Fragment main()
{
	Fragment output;
	output.color = use(fooBuffer[42].v);
	output.extra = use(fooBuffer[999].v);
	return output;
}