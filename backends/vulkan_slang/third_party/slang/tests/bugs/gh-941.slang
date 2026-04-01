//TEST:SIMPLE(filecheck=CHECK): -profile spirv_1_5+fragment -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=GLSL): -profile spirv_1_5+fragment -entry main -target glsl

// Ensure that we add the `GL_EXT_nonuniform_qualifier` extension for any code that uses unbounded-size arrays of resources.

// CHECK: OpCapability RuntimeDescriptorArray
// GLSL: #extension GL_EXT_nonuniform_qualifier : require

Texture2D t[];
SamplerState s;

cbuffer C
{
	float2 uv;
	uint index;
}

float4 main() : SV_Target
{
	return t[index].Sample(s, uv);
}
