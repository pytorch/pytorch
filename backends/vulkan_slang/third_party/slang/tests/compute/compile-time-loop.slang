//TEST(compute):COMPARE_RENDER_COMPUTE: -shaderobj
//TEST(compute):COMPARE_RENDER_COMPUTE: -mtl -shaderobj

//TEST_INPUT: Texture2D(size=4, content = one):name t
//TEST_INPUT: Sampler:name s
//TEST_INPUT: ubuffer(data=[0], stride=4):out, name outputBuffer

Texture2D t;
SamplerState s;
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

[shader("vertex")]
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

[shader("fragment")]
FragmentStageOutput fragmentMain(FragmentStageInput input)
{
	FragmentStageOutput output;

	float3 color = input.coarseVertex.color;
    float2 uv = input.coarseVertex.uv;
	output.fragment.color = float4(color, 1.0);


	float4 result = 0;
	$for(i in Range(0,5))
	{
		float4 v = t.Sample(s, uv, int2(i - 2, 0));
		result += v;
	}

	outputBuffer[0] = result.x;

	return output;
}
