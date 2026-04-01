#version 450
//TEST_DISABLED(smoke):REFLECTION:-profile ps_4_0 -target glsl

// Note: disabled because we don't support GLSL input any more
// (kept around so we can replace with an equivalent
// test that uses HLSL input and tests GLSL layout rules)

// Confirm fix for GitHub issue #55

struct Foo
{
	vec2 f;
};

layout(set = 0, binding = 0)
buffer SomeBuffer
{
	// offset: 0, size: 4, alignment: 4
    float a;

    // offset: 16, size: 12, alignment: 16
    //
    // Note that it can't immediately follow `a`
    // bcause of its alignment
    vec3 b;

    // offset: 28, size: 16, alignment: 4, stride: 4
    //
    // This array can be densely packed and 4-byte aligned under `std430` rules
    float c[4];

    // offset: 48, size: 8, alignment: 8
    //
    // This nees to be bumped up to a 8-byte aligned boundary
    vec2 d;

    // offset: 56, size: 8, alignment: 8
    //
    // This can come right after `d`, because `struct` types no longer
    // get an artificial padding out to 16-byte alignment 
    Foo e;

    // offset: 64, size: 12: alignment: 16
    vec3 g;

    // offset: 76, size: 4, alignment: 4
    //
    // This can fit in the empty space after `g` because we allow
    // the size of a `vec3` to be smaller than its alignment
    float h;
};

void main()
{}
