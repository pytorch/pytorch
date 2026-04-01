//TEST(render):COMPARE_HLSL_CROSS_COMPILE_RENDER:


// This is a test to ensure that we can cross-compile a complete entry point.

float3 transformColor(float3 color)
{
	float3 result;

	result.x = sin(20.0 * (color.x + color.y));
	result.y = saturate(cos(color.z * 30.0));
	result.z = sin(color.x * color.y * color.z * 100.0);

	result = 0.5 * (result + 1);

	return result;
}

cbuffer Uniforms
{
	float4x4 modelViewProjection;
};

struct AssembledVertex
{
	float3	position;
	float3	color;
};

struct CoarseVertex
{
	float3	color;
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

	color = transformColor(color);

	output.fragment.color = float4(color, 1.0);

	return output;
}
