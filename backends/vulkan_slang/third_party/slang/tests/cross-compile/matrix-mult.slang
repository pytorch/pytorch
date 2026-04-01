//TEST_DISABLED:CROSS_COMPILE: -profile ps_5_0 -entry main -target spirv-assembly

// Confirm that order of arguments to matrix-vector
// multiplication gets reversed when generating GLSL.

cbuffer C
{
	float4x3 m;
};

float4 main(float3 v) : SV_Target
{
	return mul(m, v);
}
