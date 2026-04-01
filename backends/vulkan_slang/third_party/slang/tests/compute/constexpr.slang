// constexpr.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -gcompute -shaderobj
//DISABLED://TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -gcompute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-mtl -gcompute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -gcompute -shaderobj

//TEST_INPUT: Texture2D(size=4, content = one):name tex
//TEST_INPUT: Sampler:name samp
//TEST_INPUT: ubuffer(data=[0 0], stride=4):out,name outputBuffer

// Note: Vulkan version of this test is disabled pending adding
// support for rendering tests to the harness.

Texture2D tex;
SamplerState samp;
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
	// The texel offset argument to `Texture2D.Sample` is
	// required to be `constexpr`. This test is going to
	// check that we correctly propagate this constraint
	// backward to the value `a`.
	//
	// Because the HLSL compiler(s) already do this kind
	// of propagation, the only real way to test this
	// will be to target Vulkan, where the standard
	// GLSL compiler gives an error message rather than
	// infer `const`-ness.

	uint a = 0;
	constexpr uint b = 1;

	uint2 ab = uint2(a,b);

	FragmentStageOutput output;

	float3 color = input.coarseVertex.color;
    float2 uv = input.coarseVertex.uv;
	output.fragment.color = float4(color, 1.0);

	float4 val = float4(color, 1.0);
	val = val - 16*tex.Sample(samp, uv, int2(ab));

	outputBuffer[0] = 1;

	if(val.x < 0)
		discard;

	outputBuffer[1] = 1;
	return output;
}
