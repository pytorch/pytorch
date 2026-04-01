// component-write.slang

// This tests that writing to individual components of the output struct works

//TEST:CROSS_COMPILE:-target spirv -profile glsl_450+GL_EXT_mesh_shader -entry main -stage mesh

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
    const uint numVertices = 3;
    const uint numPrimitives = 1;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if(tig < numVertices) {
        verts[tig].pos = float4(positions[tig], 0, 1);
        verts[tig].color = colors[tig];
    }

    if(tig < numPrimitives) {
        triangles[tig] = uint3(0,1,2);
    }
}

