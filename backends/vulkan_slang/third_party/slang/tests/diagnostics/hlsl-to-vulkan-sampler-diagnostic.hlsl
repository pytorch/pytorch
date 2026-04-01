//DIAGNOSTIC_TEST:SIMPLE:-target glsl -profile ps_4_0 -entry main -fvk-t-shift 5 all -fvk-t-shift 7 2  -fvk-s-shift -3 0 -fvk-u-shift 1 2 -no-codegen
//DIAGNOSTIC_TEST:SIMPLE:-target glsl -profile ps_4_0 -entry main -no-codegen

// This tests that combined texture sampler objects which have D3D style register assignments, but no vk::binding,
// show an appropriate warning.
// The warning should appear even if the user used -fvk-xxx-shift options, because those options do not serve to map
// two register assignments of different types into a single vulkan binding.

struct Data
{
    float a;
    int b;
};

// Neither vk::binding, nor register, no warning
Sampler2D 		cs0;

// Only vk::binding, no warning
[[vk::binding(0,0)]]
Sampler2D 		cs1;

// Both vk::binding and register, no warning
[[vk::binding(1,0)]]
Sampler2D 		cs2 : register(s0): register(t0);

// Only register, should warn without recommending vk-xxx-shift, since that would not help map 2 d3d registers to one vk binding.
Sampler2D 		cs3 : register(s1): register(t1);

float4 main() : SV_TARGET
{
	return float4(1, 1, 1, 0);
}
