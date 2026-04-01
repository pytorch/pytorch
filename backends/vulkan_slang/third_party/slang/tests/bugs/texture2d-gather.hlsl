//TEST(smoke):COMPARE_HLSL_RENDER:
//DISABLE_TEST(smoke):COMPARE_HLSL_RENDER:-mtl
//TEST_INPUT: Texture2D(size=16, content=chessboard, format=R32Float):name g_texture
//TEST_INPUT: Sampler :name g_sampler

Texture2D<float> g_texture;

SamplerState g_sampler;

cbuffer Uniforms
{
	float4x4 modelViewProjection;
}

struct AssembledVertex
{
	float3	position;
	float3	color;
	float2	uv;
};

// Vertex  Shader
struct VertexStageInput
{
	AssembledVertex assembledVertex	: A;
};

struct VertexStageOutput
{
	float4          color           : COLOR; 
	float4			position		: SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(VertexStageInput input) 
{
    VertexStageOutput output;
    
    output.position = mul(modelViewProjection, float4(input.assembledVertex.position, 1.0));
    output.color = float4(input.assembledVertex.color, 1.0f);
    
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput input) : SV_Target
{
    return g_texture.GatherRed(g_sampler, input.color.xy);
}