//TEST:SIMPLE:
// test that `struct` decls work

// Note(tfoley): disabled during syntax transition
// #include "pipeline-simple.slang.h"

// struct declaration
struct Foo
{
	float3 a;
	float3 b;
};

// function on a struct
Foo makeFoo(float x, float y)
{
	// local of struct type
	Foo foo;
	foo.a = float3(x);
	foo.b = float3(y);
	return foo;
}

/* Note(tfoley): disabled during syntax transition

template shader Test()
//    targets StandardPipeline
{
	// Uniform of struct type
	param Foo foo1;

    @MeshVertex float3 position;
    @MeshVertex float3 color;

    param mat4 modelViewProjection;

    public vec4 projCoord = modelViewProjection * vec4(position, 1.0);

    // Component of struct type
    // Note(tfoley): use of `public` here required to work around parser limitations
    public Foo foo2 = makeFoo(color.x, color.y);

    //
    float3 result = foo1.a + foo2.b;

    out @Fragment vec4 colorTarget = vec4(result,1);
}

*/