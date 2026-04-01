//TEST(compute, vulkan):COMPARE_RENDER_COMPUTE:-vk -shaderobj -render-feature shader-resource-min-lod
//TEST(compute):COMPARE_RENDER_COMPUTE:-shaderobj -render-feature shader-resource-min-lod
// WGSL: There is no SampleGrad taking a LOD-clamp parameter in

//TEST_INPUT: Texture2D(size=4, content = one):name=t2D
//TEST_INPUT: Sampler:name=samplerState
//TEST_INPUT: ubuffer(data=[0], stride=4):out,name=outputBuffer

Texture2D  t2D;
SamplerState samplerState;
RWStructuredBuffer<float> outputBuffer;

cbuffer Uniforms
{
	float4x4 modelViewProjection;
}

struct AssembledVertex
{
	float3	position;
	float3	color;
    float2  uv;
};

struct CoarseVertex
{
	float3	color;
    float2  uv;
};

struct Fragment
{
	float4 color;
};


// Vertex  Shader

struct VertexStageInput
{
	AssembledVertex assembledVertex	: A;
};

struct VertexStageOutput
{
	CoarseVertex	coarseVertex	: CoarseVertex;
	float4			sv_position		: SV_Position;
};

VertexStageOutput vertexMain(VertexStageInput input)
{
	VertexStageOutput output;

	float3 position = input.assembledVertex.position;
	float3 color	= input.assembledVertex.color;

	output.coarseVertex.color = color;
	output.sv_position = mul(modelViewProjection, float4(position, 1.0));
    output.coarseVertex.uv = input.assembledVertex.uv;
	return output;
}

// Fragment Shader

struct FragmentStageInput
{
	CoarseVertex	coarseVertex	: CoarseVertex;
};

struct FragmentStageOutput
{
	Fragment fragment	: SV_Target;
};

FragmentStageOutput fragmentMain(FragmentStageInput input)
{
	FragmentStageOutput output;

	float3 color = input.coarseVertex.color;
    float2 uv = input.coarseVertex.uv;
	output.fragment.color = float4(color, 1.0);

    float4 val = t2D.SampleGrad(samplerState, uv, float2(0,0), float2(0,0), 0, 0.5f);   
    outputBuffer[0] = val.x;
	return output;
}
