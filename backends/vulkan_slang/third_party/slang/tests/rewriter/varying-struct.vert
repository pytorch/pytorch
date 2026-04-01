#version 450 core
//TEST_DISABLED:COMPARE_GLSL:

#if defined(__SLANG__)

__import varying_struct;

VS_OUT main(VS_IN foo)
{
    return doIt(foo);
}

#else

struct VS_IN
{
	vec4 x;
	vec4 y;	
};

struct VS_OUT
{
	vec4 color;
	vec4 posH;
};

VS_OUT doIt(VS_IN i)
{
	VS_OUT o;
	o.color = i.x;
	o.posH = i.y;
	return o;
}

layout(location = 0)
out vec4 SLANG_out_bar_color;

layout(location = 0)
in vec4 SLANG_in_foo_x;

layout(location = 1)
in vec4 SLANG_in_foo_y;

void main()
{
	VS_OUT SLANG_tmp_0 = doIt(VS_IN(SLANG_in_foo_x, SLANG_in_foo_y));
	SLANG_out_bar_color = SLANG_tmp_0.color;
	gl_Position = SLANG_tmp_0.posH;
}

#endif
