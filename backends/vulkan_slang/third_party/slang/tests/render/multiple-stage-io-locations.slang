//TEST(smoke,render):COMPARE_HLSL_RENDER:
// TODO: Investigate Metal failure
//DISABLE_TEST(smoke,render):COMPARE_HLSL_RENDER: -mtl

cbuffer Uniforms
{
    float4x4 modelViewProjection;
}

struct AssembledVertex
{
    float3 position;
    float3 color;
};

struct Fragment
{
    float4 color;
};

// Vertex  Shader

struct VertexStageInput
{
    AssembledVertex assembledVertex : A;
};

struct VertexStageOutput
{
    float3 color           : VERTEX_COLOR;
    float3 localPosition   : VERTEX_LOCAL_POSITION;
    float4 sv_position     : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(VertexStageInput input)
{
    VertexStageOutput output;

    float3 position = input.assembledVertex.position;
    float3 color    = input.assembledVertex.color;

    output.color         = color;
    output.sv_position   = mul(modelViewProjection, float4(position, 1.0));
    output.localPosition = position;

    return output;
}

// Fragment Shader

struct FragmentStageInput
{
    float3 color         : VERTEX_COLOR;
    float3 localPosition : VERTEX_LOCAL_POSITION;
};

struct FragmentStageOutput
{
    Fragment fragment : SV_Target;
};

[shader("fragment")]
FragmentStageOutput fragmentMain(FragmentStageInput input)
{
    FragmentStageOutput output;

    float3 color = input.color;

    if (input.color.y < input.color.z)
    {
        output.fragment.color = float4(input.localPosition, 1.0);
    }
    else
    {
        output.fragment.color = float4(input.color, 1.0);
    }

    return output;
}
