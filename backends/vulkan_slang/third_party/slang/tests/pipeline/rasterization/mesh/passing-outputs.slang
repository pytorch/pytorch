// component-write.slang

// This tests that writing to individual components of the output struct works

//TEST:SIMPLE(filecheck=SPIRV): -target spirv-assembly -entry main -stage mesh -profile glsl_450+spirv_1_4

// DXC is stricter than we are about passing references to individual mesh shader outputs
// We could get around this by doing what we do for GLSL, i.e. use a temporary
// variable to pass as the out parameter, and then copy that into the array
// after the function call.
//TEST_DISABLED:CROSS_COMPILE:-target dxil-assembly  -entry main -stage mesh -profile sm_6_6

// SPIRV: OpEntryPoint MeshEXT %main

const static uint MAX_VERTS = 3;
const static uint MAX_PRIMS = 1;

struct Texes
{
    float2 tex1;
    float4 tex2;
}

struct Vertex
{
    float4 pos : SV_Position;
    float3 col : Color;
    Texes  ts  : Coord;
};

void everything<let N : uint>(OutputVertices<Vertex, N> vs)
{
    vs[0] = {float4(0), float3(1)};
}

void just_one(out Vertex v)
{
    v = {float4(0), float3(1)};
}

void just_two(out Vertex v, out Vertex w)
{
    v = {float4(0), float3(1)};
    w = v;
}

void part_of_one(out float4 p)
{
    p = float4(1,2,3,4);
}

void write_struct(out Texes t)
{
    t.tex1 = float2(0);
    t.tex2 = float4(1);
}

// Split out the things to test to avoid main becoming an unreadable jumble
void a<let N : uint>(OutputVertices<Vertex, N> vs)
{
    // Test passing a reference to the entire array
    everything(vs);
}

void b<let N : uint>(OutputVertices<Vertex, N> vs)
{
    // test passing two references to the same element
    just_two(vs[0], vs[0]);
}

void c<let N : uint>(OutputVertices<Vertex, N> vs, uint tig)
{
    // Test passing a reference to an element
    just_one(vs[tig]);
}

void d<let N : uint>(OutputVertices<Vertex, N> vs, uint tig)
{
    // Test passing references to different elements (to check that the operand
    // rewriting doesn't mess the order)
    just_two(vs[tig], vs[0]);
}

void e<let N : uint>(OutputVertices<Vertex, N> vs, uint tig)
{
    // Test passing a scalar member and a struct member and a struct member's member
    part_of_one(vs[tig].pos);
    write_struct(vs[tig].ts);
    part_of_one(vs[tig].ts.tex2);
}

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main(
    in uint tig : SV_GroupIndex,
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputIndices<uint3, MAX_PRIMS> triangles
    )
{
    const uint numVertices = 3;
    const uint numPrimitives = 1;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if(tig < numVertices) {
        a(verts);
        b(verts);
        c(verts, tig);
        d(verts, tig);
        e(verts, tig);
    }

    if(tig < numPrimitives) {
        triangles[tig] = uint3(0,1,2);
    }
}

