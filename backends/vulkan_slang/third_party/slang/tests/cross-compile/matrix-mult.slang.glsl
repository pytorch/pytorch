//TEST_IGNORE_FILE:
#version 420

layout(binding = 0)
uniform C
{
	mat4x3 m;
};

vec4 main_(vec3 v)
{
	return v * m;
}

layout(location = 0)
in vec3 SLANG_in_v;

layout(location = 0)
out vec4 SLANG_out_main_result;

void main()
{
	vec3 v = SLANG_in_v;
	vec4 main_result = main_(v);
	SLANG_out_main_result = main_result;
}
