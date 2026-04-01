// This shader is purposefully designed to be so slow it will cause a GPU timeout/crash.

// Uniform data to be passed from application -> shader.
cbuffer Uniforms
{
    float4x4 modelViewProjection;

    // We want to make things fail so we can get an aftermath capture, 
    // so lets have a count that makes things really slow.
    int failCount;
}

// Per-vertex attributes to be assembled from bound vertex buffers.
struct AssembledVertex
{
    float3	position : POSITION;
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

    float3 position = assembledVertex.position;
    float3 color    = assembledVertex.color;

    output.coarseVertex.color = color;
    output.sv_position = mul(modelViewProjection, float4(position, 1.0));

    return output;
}

// Fragment Shader

[shader("fragment")]
float4 fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex) : SV_Target
{
    float3 color = coarseVertex.color;

    float factor = 0.0f;

    // Waste lots of cycles
    for (int i = 0; i < failCount; ++i)
    {
        factor += 1.0e-20 * sin(float(i & 0xffff));
        factor += 1.0e-21 * cos(float(i & 0xfff) + 1.0);
        factor += 1.0e-8f * tan(float(i & 0xfffff));
    }

    factor = abs(factor);

    while (factor < 0.25)
    {
        factor += factor;
    }

    return float4(color, 1.0) * factor;
}
