//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK): -task -output-using-type -dx12 -use-dxil -profile sm_6_6 -render-features mesh-shader
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK): -task -output-using-type -vk -profile sm_6_5 -render-features mesh-shader

// Similar to task-simple, except that the payload is declared as a groupshared
// variable. During lowerin to GLSL and SPIR-V we'll have to identify this as
// the variable being passed to DispatchMesh and emit it using the
// taskPayloadSharedEXT rate.

// CHECK:      0
// CHECK-NEXT: 1
// CHECK-NEXT: 8
// CHECK-NEXT: 27

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer

RWStructuredBuffer<float> outputBuffer;

cbuffer Uniforms
{
    float4x4 modelViewProjection;
}

//
// Task shader
//

struct MeshPayload
{
    int exponent;
};

groupshared MeshPayload p;

[numthreads(1, 1, 1)]
[shader("amplification")]
void taskMain(in uint tig : SV_GroupIndex)
{
    p.exponent = 3;
    DispatchMesh(1,1,1,p);
}


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

struct Vertex
{
  float4 pos : SV_Position;
  float3 color : Color;
  int index : Index;
  int value : Value;
};

const static uint MAX_VERTS = 12;
const static uint MAX_PRIMS = 4;

[outputtopology("triangle")]
[numthreads(12, 1, 1)]
void meshMain(
    in uint tig : SV_GroupIndex,
    in payload MeshPayload meshPayload,
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputIndices<uint3, MAX_PRIMS> triangles)
{
    const uint numVertices = 12;
    const uint numPrimitives = 4;
    SetMeshOutputCounts(numVertices, numPrimitives);

    if(tig < numVertices)
    {
        const int tri = tig / 3;
        verts[tig] = {float4(positions[tig % 3], 0, 1), colors[tig % 3], tri, int(pow(tri, meshPayload.exponent))};
    }

    if(tig < numPrimitives)
        triangles[tig] = tig * 3 + uint3(0,1,2);
}

//
// Fragment Shader
//

struct Fragment
{
    float4 color : SV_Target;
};

Fragment fragmentMain(Vertex input)
{
    outputBuffer[input.index] = input.value;

    Fragment output;
    output.color = float4(input.color, 1.0);
    return output;
}

