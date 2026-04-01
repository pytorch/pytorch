// graphics-smoke.slang

// Per-vertex attributes to be assembled from bound vertex buffers.
struct AssembledVertex
{
    float3	positionA : POSITIONA;
    float3  positionB : POSITIONB;
    float3	color    : COLOR;
};

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 color;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};

// Vertex  Shader

struct VertexStageOutput
{
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.positionA + assembledVertex.positionB;
    float3 color    = assembledVertex.color;

    output.coarseVertex.color = color;
    output.sv_position = float4(position, 1.0);

    return output;
}

// Fragment Shader

[shader("fragment")]
float4 fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex) : SV_Target
{
    float3 color = coarseVertex.color;

    return float4(color, 1.0);
}
