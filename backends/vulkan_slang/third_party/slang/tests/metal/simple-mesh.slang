//TEST:SIMPLE(filecheck=METAL): -entry meshMain -stage mesh -target metal

//
// Mesh shader
//

const static float2 positions[3] = {
    float2(0.0, -0.5),
    float2(0.5, 0.5),
    float2(-0.5, 0.5)
};

const static float3 colors[3] = {
    float3(1.0, 1.0, 0.0),
    float3(0.0, 1.0, 1.0),
    float3(1.0, 0.0, 1.0)
};

struct MeshPayload
{
    int exponent;
};


struct Vertex
{
    float4 pos : SV_Position;
    float3 color : Color;
    int index : Index;
    int value : Value;
};

struct Primitive
{
    uint prim : SV_PrimitiveID;
};

const static uint MAX_VERTS = 12;
const static uint MAX_PRIMS = 4;

[outputtopology("triangle")]
[numthreads(12, 1, 1)]
void meshMain(
    in uint tig: SV_GroupIndex,
    in payload MeshPayload meshPayload,
    // METAL: const MeshPayload_0 object_data* meshPayload_0
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputIndices<uint3, MAX_PRIMS> triangles,
    OutputPrimitives<Primitive, MAX_PRIMS> primitives
    )
{
    const uint numVertices = 12;
    const uint numPrimitives = 4;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if (tig < numVertices)
    {
        const int tri = tig / 3;
        verts[tig] = { float4(positions[tig % 3], 0, 1), colors[tig % 3], tri, int(pow(tri, meshPayload.exponent)) };
    }

    if (tig < numPrimitives)
    {
        // METAL: _slang_mesh.set_index({{.*}}+0,{{.*}}[0]);
        // METAL: _slang_mesh.set_index({{.*}}+1,{{.*}}[1]);
        // METAL: _slang_mesh.set_index({{.*}}+2,{{.*}}[2]);
        triangles[tig] = tig * 3 + uint3(0, 1, 2);
        // METAL: _slang_mesh.set_primitive({{.*}}
        primitives[tig] = { tig };
    }
}
