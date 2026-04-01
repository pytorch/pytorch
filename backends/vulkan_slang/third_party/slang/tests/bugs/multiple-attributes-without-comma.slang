// multiple-attributes-without-comma.slang

// Test that `,` is not need between attributes 

//TEST:SIMPLE(filecheck=CHECK):-target glsl -entry main -stage fragment -line-directive-mode none

//CHECK: layout(location = 0, index = 1)

struct FragmentOutput
{
	[[vk::location(0)]]
	float4 a : SV_Target0;

	[[vk::location(0) vk::index(1)]]
	float4 b : SV_Target1;
}

[shader("fragment")]
FragmentOutput main(float4 v : V)
{
	FragmentOutput f = {};
	f.a = v;
	f.b = v;
	return f;	
}
