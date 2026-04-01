//TEST_IGNORE_FILE:

// We are declaring the uniform shader parameters
// in this imported file, to ensure that they are
// emitted correctly.

cbuffer Uniforms
{
	float4x4 modelViewProjection;
};

float3 transformColor(float3 color)
{
	float3 result;

	result.x = sin(20.0 * (color.x + color.y));
	result.y = saturate(cos(color.z * 30.0));
	result.z = sin(color.x * color.y * color.z * 100.0);

	result = 0.5 * (result + 1);

	return result;
}