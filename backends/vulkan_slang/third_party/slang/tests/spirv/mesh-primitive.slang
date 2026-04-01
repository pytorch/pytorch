//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -skip-spirv-validation

// CHECK: OpDecorate %primitives_color Location 0
// CHECK: OpDecorate %primitives_color PerPrimitive
// CHECK: OpDecorate %prim_color Location 0
// CHECK: OpDecorate %prim_color Flat

const static uint MAX_VERTS = 6;
const static uint MAX_PRIMS = 2;

const static float2 positions[MAX_VERTS] = {
  float2(0.0, -0.5),
  float2(0.5, 0),
  float2(-0.5, 0),
  float2(0.0, 0.5),
  float2(0.5, 0),
  float2(-0.5, 0),
};

struct Vertex
{
  float4 pos : SV_Position;
};

struct Primitive
{
  [[vk::location(0)]] float3 color;
}

[outputtopology("triangle")]
[numthreads(MAX_VERTS, 1, 1)]
[shader("mesh")]
void entry_mesh(
    in uint tig : SV_GroupThreadID,
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputIndices<uint3, MAX_PRIMS> triangles,
    OutputPrimitives<Primitive, MAX_PRIMS> primitives)
{
    const uint numVertices = MAX_VERTS;
    const uint numPrimitives = MAX_PRIMS;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if(tig < numVertices) {
        verts[tig] = {float4(positions[tig], 0, 1)};
    }

    if(tig < numPrimitives) {
        triangles[tig] = uint3(0,1,2) + tig * 3;
        primitives[tig] = { float3(1,0,0) };
    }
}

struct FragmentOut
{
    [[vk::location(0)]] float3 color;
};

[shader("fragment")]
FragmentOut entry_fragment(in nointerpolation Primitive prim)
{
    FragmentOut frag_out;
    frag_out.color = prim.color;
    return frag_out;
}
