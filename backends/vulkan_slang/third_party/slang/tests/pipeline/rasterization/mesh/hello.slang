// hello.slang

// Test that a simple mesh shader compiles

//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage mesh -profile glsl_450+spirv_1_4
//TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly  -entry main -stage mesh -profile sm_6_6

// DXIL: call void @dx.op.setMeshOutputCounts
// DXIL: call void @dx.op.storeVertexOutput.f32
// DXIL: call void @dx.op.emitIndices
// SPIRV: OpEntryPoint MeshEXT %main

// SPIRV-DAG: OpExecutionMode %main OutputVertices 3
// SPIRV-DAG: OpExecutionMode %main OutputPrimitives{{NV|EXT}} 1
// SPIRV-DAG: OpExecutionMode %main OutputTriangles{{NV|EXT}}

// SPIRV: OpSetMeshOutputsEXT

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
        verts[tig] = {float4(positions[tig], 0, 1), colors[tig]};
    }

    if(tig < numPrimitives) {
        triangles[tig] = uint3(0,1,2);
    }
}

