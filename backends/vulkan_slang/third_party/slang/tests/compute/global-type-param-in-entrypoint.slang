// tests/compute/*type-param* tests are broken #5627
//DISABLED_TEST(compute):COMPARE_RENDER_COMPUTE: -shaderobj

//TEST_INPUT: cbuffer(data=[1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0]):name Uniforms
//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
//TEST_INPUT: global_type VertImpl

interface IVertInterpolant
{
	float4 getColor();
}

type_param TVertInterpolant : IVertInterpolant; 

struct VertImpl : IVertInterpolant
{
	float3 color;
	float4 getColor()
	{
		return float4(1.0);
	}
};

RWStructuredBuffer<float> outputBuffer;

cbuffer Uniforms
{
	float4x4 modelViewProjection;
}

struct AssembledVertex
{
	float3	position;
	TVertInterpolant interpolants;
    float2  uv;
};

struct CoarseVertex
{
	TVertInterpolant	interpolants;
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
	output.coarseVertex.interpolants = input.assembledVertex.interpolants;
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

	float4 color = input.coarseVertex.interpolants.getColor();
    float2 uv = input.coarseVertex.uv;
	output.fragment.color = color;
    outputBuffer[0] = color.x;
    outputBuffer[1] = color.y;
    outputBuffer[2] = color.z;
    outputBuffer[3] = color.w;
	return output;
}