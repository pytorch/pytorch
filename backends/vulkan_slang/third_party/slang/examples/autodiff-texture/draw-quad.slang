// Vertex and fragment shader to draw a textured quad on screen.

cbuffer Uniforms
{
    int x;
    int y;
    int width;
    int height;
    int viewWidth;
    int viewHeight;
    Texture2D texture;
    SamplerState sampler;
}

struct AssembledVertex
{
    float3	position : POSITION;
};

struct Fragment
{
    float4 color;
};

struct VertexStageOutput
{
    float2 uv : UV;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;

    output.uv = position.xy;
    output.sv_position.x = (x + position.x * width) / (float)viewWidth * 2.0f - 1.0f;
    output.sv_position.y = -((y + position.y * height) / (float)viewHeight * 2.0f - 1.0f);
    output.sv_position.z = 0.5;
    output.sv_position.w = 1.0;
    return output;
}

[shader("fragment")]
float4 fragmentMain(
    float2 uv : UV) : SV_Target
{
    return float4(texture.Sample(sampler, uv).xyz, 1.0);
}
