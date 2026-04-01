//TEST:SIMPLE(filecheck=CHECK):-target glsl -profile ps_4_0 -entry main -fvk-bind-globals 5 9 -line-directive-mode none

// CHECK: layout(binding = 0)
// CHECK: uniform texture2D t_0;
// CHECK: layout(binding = 1)
// CHECK: uniform sampler sampler_0;
// CHECK: layout(binding = 5, set = 9)
// CHECK: layout(std140) uniform block_

uniform int a;
uniform float b;

Texture2D t;
SamplerState sampler;

float4 main() : SV_TARGET
{
	return t.SampleLevel(sampler, float2(a,b), 0) + float4(a, b, 1, 0);
}