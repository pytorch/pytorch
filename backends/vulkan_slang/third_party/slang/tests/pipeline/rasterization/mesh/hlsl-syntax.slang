// hlsl-syntax.slang

// Test that we can ingest hlsl mesh output syntax

//TEST:CROSS_COMPILE:-target spirv -profile sm_6_5 -entry main -stage mesh

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

// Test that we can convert the HLSL Syntax to the typed syntax
void foo(uint tig, OutputVertices<Vertex, MAX_VERTS> verts)
{
    if(tig < 3) {
        verts[tig] = {float4(positions[tig], 0, 1), colors[tig]};
    }
}

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void main(
    in uint tig : SV_GroupIndex,
    out vertices Vertex verts[MAX_VERTS],
    out indices uint3 triangles[MAX_PRIMS]
    )
{
    const uint numVertices = 3;
    const uint numPrimitives = 1;
    SetMeshOutputCounts(numVertices, numPrimitives);

    foo(tig, verts);

    if(tig < numPrimitives) {
        triangles[tig] = uint3(0,1,2);
    }
}

