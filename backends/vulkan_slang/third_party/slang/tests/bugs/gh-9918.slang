//TEST:SIMPLE(filecheck=CHECK_SPIRV): -target spirv -entry entry_mesh
//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -entry entry_mesh
//TEST:SIMPLE(filecheck=CHECK_HLSL): -target hlsl -entry entry_mesh

//CHECK_SPIRV: OpEntryPoint
//CHECK_GLSL: void main()
//CHECK_HLSL: void entry_mesh


const static float2 positions[3] = {
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5)
};

interface VertexI
{
    [mutating]
    func set_pos(float4 v);
}

interface PrimitiveI
{
    [mutating]
    func set_tint(float3 v);
}

struct Vertex : VertexI
{
    float4 pos : SV_Position;
    [mutating]
    func set_pos(float4 v) { pos = v; }
};

struct Primitive : PrimitiveI
{
    [[vk::location(0)]] nointerpolation float3 tint;
    [mutating]
    func set_tint(float3 v) { tint = v; }
}

const static uint MAX_VERTS = 3;
const static uint MAX_PRIMS = 1;

void mesh_fn<V : VertexI, P : PrimitiveI>(
    in uint tig,
    OutputIndices<uint3, MAX_PRIMS> triangles,
    OutputVertices<V, MAX_VERTS> verts,
    OutputPrimitives<P, MAX_PRIMS> primitives)
{
    const uint numVertices = 3;
    const uint numPrimitives = 1;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if (tig < numVertices) {
        V v;
        if (V is Vertex)
        {
            Vertex v2 = reinterpret<Vertex>(v);
            v2.pos = float4(positions[tig], 0, 1);
            v = reinterpret<V>(v2);
        }
        verts[tig] = v;
    }

    if (tig < numPrimitives) {
        triangles[tig] = uint3(0, 1, 2);
        primitives[tig].set_tint(float3(1, 0, 0));
    }
}

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
[shader("mesh")]
void entry_mesh(
    in uint tig: SV_GroupIndex,
    OutputIndices<uint3, MAX_PRIMS> triangles,
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputPrimitives<Primitive, MAX_PRIMS> primitives)
{
    mesh_fn(tig, triangles, verts, primitives);
}

struct FragmentOut
{
    [[vk::location(0)]] float3 color;
};

[shader("fragment")]
FragmentOut entry_fragment(in Vertex vertex, in Primitive prim)
{
    FragmentOut frag_out;
    frag_out.color = prim.tint;
    return frag_out;
}
