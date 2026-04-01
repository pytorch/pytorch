// nested-component-write.slang

// This tests that writing to individual components nested structs works for
// mesh shader outputs.

//TEST:CROSS_COMPILE:-target spirv -profile glsl_450+GL_EXT_mesh_shader -entry main -stage mesh

struct Foo
{
    float4 pos : SV_Position;
    Bar bar;
};

struct Bar
{
    Baz baz : Color;
};

struct Baz
{
    float3 color;
};

struct Vertex
{
    Foo foo;
};

const static uint MAX_VERTS = 3;
const static uint MAX_PRIMS = 1;

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main(
    in uint tig : SV_GroupIndex,
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputIndices<uint3, MAX_PRIMS> triangles
    )
{
    SetMeshOutputCounts(3, 1);

    if(tig < 3) {
        verts[tig].foo.pos = float4(0, 0, 0, 1);
        verts[tig].foo.bar.baz.color = float3(1, 2, 3);
    }

    if(tig < 1) {
        triangles[tig] = uint3(0,1,2);
    }
}

