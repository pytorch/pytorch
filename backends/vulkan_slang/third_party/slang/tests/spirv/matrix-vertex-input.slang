//TEST(compute):COMPARE_RENDER_COMPUTE(filecheck-buffer=ROWMAJOR): -vk -output-using-type
//TEST(compute):COMPARE_RENDER_COMPUTE(filecheck-buffer=ROWMAJOR): -d3d11 -output-using-type

//TEST(compute):COMPARE_RENDER_COMPUTE(filecheck-buffer=COLMAJOR): -vk -output-using-type -emit-spirv-directly -xslang -DCOLUMN_MAJOR
//TEST(compute):COMPARE_RENDER_COMPUTE(filecheck-buffer=COLMAJOR): -d3d11 -output-using-type -xslang -DCOLUMN_MAJOR

// Check that row_major and column_major matrix typed vertex input are correctly handled.

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
#ifdef COLUMN_MAJOR
    column_major float4x4 m;
#else
    row_major float4x4 m;
#endif
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

VertexStageOutput vertexMain(VertexStageInput input)
{
	VertexStageOutput output;
	output.coarseVertex.color = input.assembledVertex.m[1][2];
    output.sv_position = mul(modelViewProjection, float4(input.assembledVertex.position, 1.0));
	return output;
}

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
	output.fragment.color = float4(color, 1.0);
	outputBuffer[0] = color.x;
    // ROWMAJOR: 7.0
    // COLMAJOR: 10.0
	return output;
}
