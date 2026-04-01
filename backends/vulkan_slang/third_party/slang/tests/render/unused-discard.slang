//TEST_IGNORE_FILE:

// This file implements the "library" code
// that both the HLSL and GLSL shaders share.
//
// This code is written in Slang (more or less
// just HLSL), and will be translated as needed
// for each of the targets.

float3 transformColor(float3 color)
{
	float3 result;

	result.x = sin(20.0 * (color.x + color.y));
	result.y = saturate(cos(color.z * 30.0));
	result.z = sin(color.x * color.y * color.z * 100.0);

	result = 0.5 * (result + 1);

	return result;
}

void doConditionalDiscard(float3 color)
{
	if(color.x < 0.5)
		discard;
}
