//TEST_DISABLED(smoke):REFLECTION:-profile ps_4_0 -target glsl

// Disabled because we don't support GLSL any more.
// (kept around so we can replace with an equivalent
// test that uses HLSL input and tests GLSL layout rules)

// Confirm fix for GitHub issue #55

layout(set = 0, binding = 0)
uniform PerFrameCB
{
	vec2 offset;
	vec2 scale;
};

void main()
{}
