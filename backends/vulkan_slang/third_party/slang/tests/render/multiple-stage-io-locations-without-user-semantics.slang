// TODO: Investigate failures on non-WebGPU backends
//TEST(smoke,render):COMPARE_HLSL_RENDER: -wgpu

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
    float3 color;
    float3 localPosition;
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
    float3 color;
    float3 localPosition;
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
