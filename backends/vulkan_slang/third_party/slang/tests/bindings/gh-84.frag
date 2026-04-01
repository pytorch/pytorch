#version 450
//DISABLED_TEST:COMPARE_GLSL:

// Confirm implementation of GitHub issue #84

#if defined(__SLANG__)
#define LAYOUT(X) /* empty */
#else
#define LAYOUT(X) layout(X)
#endif

// Array of resources: should only use up one binding
LAYOUT(binding = 0)
uniform texture2D t[8];

// This should automatically get binding 1
LAYOUT(binding = 1)
uniform sampler s;

LAYOUT(binding = 2)
uniform U
{
	int i;
};

LAYOUT(location = 0)
in vec2 uv;

LAYOUT(location = 0)
out vec4 color;

void main()
{
	color = texture(sampler2D(t[i], s), uv);
}
