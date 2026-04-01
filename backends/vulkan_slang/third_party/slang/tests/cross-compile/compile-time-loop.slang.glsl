//TEST_IGNORE_FILE:
#version 420

layout(binding = 0)
uniform texture2D t;

layout(binding = 1)
uniform sampler s;

vec4 main_(vec2 uv)
{
	vec4 result = vec4(0);

	result += textureOffset(sampler2D(t,s), uv, ivec2(0 - 2, 0));
	result += textureOffset(sampler2D(t,s), uv, ivec2(1 - 2, 0));
	result += textureOffset(sampler2D(t,s), uv, ivec2(2 - 2, 0));
	result += textureOffset(sampler2D(t,s), uv, ivec2(3 - 2, 0));
	result += textureOffset(sampler2D(t,s), uv, ivec2(4 - 2, 0));

	return result;
}

layout(location = 0)
in vec2 SLANG_in_uv;

layout(location = 0)
out vec4 SLANG_out_main_result;

void main()
{
	vec2 uv = SLANG_in_uv;
	vec4 main_result = main_(uv);
	SLANG_out_main_result = main_result;
}