// Disabled because Slang compiler doesn't support GLSL as an input language
//TEST_IGNORE_FILE
//TEST:COMPARE_GLSL:-profile glsl_fragment_450
// matrix-mult.glsl
#version 450

// Confirm that we don't exchange the operands
// to a matrix-vector multiply when we recognize
// one in "raw" GLSL code.

#ifdef __SLANG__
__import empty;
#endif

layout(binding = 0)
uniform C
{
	mat4x4 m;
};

layout(location = 0)
in vec4 v;

layout(location = 0)
out vec4 r;

void main()
{
	r = m * v;
}
