#version 450
//TEST_DISABLED:COMPARE_GLSL:

// Test workaround for glslang issue #988
// (https://github.com/KhronosGroup/glslang/issues/988)


#if defined(__SLANG__)


__import glslang_bug_988_workaround;

uniform U
{
	Foo foo;	
};

vec4 doIt(Foo foo)
{
	return foo.bar;
}

layout(location = 0)
out vec4 result;

void main()
{
	vec4 r = doIt(foo);
	result = r;
}

#else

struct Foo
{
	vec4 bar;
};

layout(binding = 0)
uniform U
{
	Foo foo;	
};

vec4 doIt(Foo foo)
{
	return foo.bar;
}

layout(location = 0)
out vec4 result;

void main()
{
	Foo SLANG_tmp_0 = foo;
	vec4 r = doIt(SLANG_tmp_0);
	result = r;
}

#endif
