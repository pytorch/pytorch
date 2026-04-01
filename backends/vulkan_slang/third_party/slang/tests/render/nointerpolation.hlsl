//TEST(smoke):COMPARE_HLSL_RENDER:
// TODO: Investigate Metal failure
//DISABLE_TEST(smoke):COMPARE_HLSL_RENDER: -mtl

// Confirm that the `nointerpolation` modifier
// makes it through Slang codegen with the
// same effect as for HLSL.

cbuffer Uniforms
{
	float4x4 modelViewProjection;
}

struct AssembledVertex
{
	float3	position;
	float3	color;
};

struct CoarseVertex
{
	nointerpolation float3	color;
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

	output.fragment.color = float4(color, 1.0);

	return output;
}

