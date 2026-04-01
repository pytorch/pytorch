//TEST:COMPARE_HLSL: -profile ps_4_0 -entry main

// Confirm that the `-split-mixed-types` flag works.

#ifdef __SLANG__

// HLSL input:
//
// - Uses at least one `import` of Slang code
// - Uses an aggregate type that mixes resource and non-resource types
//

__import type_splitting;

struct Foo
{
	Texture2D 		t;
	SamplerState 	s;
	float2 			u;
};

cbuffer C
{
	Foo foo;
}

float4 main() : SV_Target
{
	return foo.t.Sample(foo.s, foo.u);	
}

#else

// Equivalent raw HLSL:
//
// - Fields of resource type have been stripped from original type definition
// - Fields of resource type get hoisted out of variable declarations
//

struct Foo_0
{
	float2 u_0;
};

struct SLANG_ParameterGroup_C_0
{
	Foo_0 foo_0;	
};

cbuffer C_0
{
	SLANG_ParameterGroup_C_0 C_0;
}

Texture2D    C_foo_t_0;
SamplerState C_foo_s_0;

float4 main() : SV_TARGET
{
	return C_foo_t_0.Sample(C_foo_s_0, C_0.foo_0.u_0);	
}

#endif

