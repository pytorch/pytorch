#version 450
//TEST_IGNORE_FILE:

layout(binding = 0)
uniform sampler SLANG_hack_samplerForTexelFetch;

layout(binding = 1)
uniform texture2D t;

vec4 main_()
{
	uint x, y, mipCount;

	x = textureSize(sampler2D(t,SLANG_hack_samplerForTexelFetch), 0).x;
	y = textureSize(sampler2D(t,SLANG_hack_samplerForTexelFetch), 0).y;
	mipCount = textureQueryLevels(sampler2D(t,SLANG_hack_samplerForTexelFetch));

    return vec4(
    	float(x),
    	float(y),
    	float(mipCount),
    	0.0);
}

layout(location = 0)
out vec4 SLANG_out_main_result;

void main()
{
	vec4 main_result = main_();
	SLANG_out_main_result = main_result;
}
