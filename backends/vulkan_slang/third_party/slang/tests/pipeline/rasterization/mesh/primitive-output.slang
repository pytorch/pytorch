// hello.slang

// Test that a simple mesh shader compiles

//TEST:SIMPLE(filecheck=SPV_CHECK):-target spirv -profile glsl_450+spirv_1_4 -entry main -stage mesh -skip-spirv-validation

//SPV_CHECK-NOT: SPV_NV_mesh_shader
//SPV_CHECK: SPV_EXT_mesh_shader
//SPV_CHECK: CullPrimitiveEXT

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

struct Vertex
{
    float4 pos : SV_Position;
    float3 color : Color;
};

struct Prim
{
    float3 triangleNormal : Normal;
    uint id : SV_PrimitiveID;
    bool cull : SV_CullPrimitive;
};

const static uint MAX_VERTS = 3;
const static uint MAX_PRIMS = 1;

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main(
    in uint tig : SV_GroupIndex,
    OutputIndices<uint3, MAX_PRIMS> triangles,
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputPrimitives<Prim, MAX_PRIMS> primitives
    )
{
    const uint numVertices = 3;
    const uint numPrimitives = 1;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if(tig < numVertices) {
        verts[tig] = {float4(positions[tig], 0, 1), colors[tig]};
    }

    if(tig < numPrimitives) {
        triangles[tig] = uint3(0,1,2);
        primitives[tig] = {float3(0,0,1), tig, false};
    }
}

