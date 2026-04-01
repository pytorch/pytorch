//TEST:SIMPLE(filecheck=GLSL): -profile spirv_1_5+fragment -entry fragmentMain -target glsl

// Ensure that the use of combined samplers does not needlessly require
// GL_EXT_samplerless_texture_functions

// GLSL-NOT: #extension GL_EXT_samplerless_texture_functions : require

layout(binding=1) Sampler3D sampler;

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
	int sizeX;
	int sizeY;
	int sizeZ;
	sampler.GetDimensions(sizeX, sizeY, sizeZ);
	return float4(sizeX, sizeY, sizeZ, 1.0);
}
