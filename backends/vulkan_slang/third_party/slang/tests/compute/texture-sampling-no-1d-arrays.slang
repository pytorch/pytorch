//TEST(compute):COMPARE_RENDER_COMPUTE: -shaderobj -output-using-type
//TEST(compute):COMPARE_RENDER_COMPUTE: -shaderobj -output-using-type -vk
//TEST(compute):COMPARE_RENDER_COMPUTE: -shaderobj -output-using-type -mtl


// WebGPU only supports 1 MIP level for 1d textures
// https://www.w3.org/TR/webgpu/#abstract-opdef-maximum-miplevel-count
//TEST_INPUT: Texture1D(size=4, content = one, mipMaps=1):name=t1D
//TEST_INPUT: Texture2D(size=4, content = one):name=t2D
//TEST_INPUT: Texture3D(size=4, content = one):name=t3D
//TEST_INPUT: TextureCube(size=4, content = one):name=tCube
//TEST_INPUT: Texture2D(size=4, content = one, arrayLength=2):name=t2dArray
//TEST_INPUT: TextureCube(size=4, content = one, arrayLength=2):name=tCubeArray
//TEST_INPUT: Sampler:name=samplerState
//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer

Texture1D t1D;
Texture2D t2D;
Texture3D t3D;
TextureCube tCube;
Texture2DArray t2dArray;
TextureCubeArray tCubeArray;
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

    float4 val = 0.0;
    val += t1D.Sample(samplerState, uv.x);
    val += t2D.Sample(samplerState, uv);
    val += t3D.Sample (samplerState, float3(uv, 0.5));
    
    val += t2dArray.Sample(samplerState, float3(uv, 0.0));
    
    val += tCubeArray.Sample(samplerState, float4(uv, 0.5, 0.0));
    val += tCube.Sample(samplerState, float3(uv, 0.5));

    val += t2D.Load(int3(0));
    val += t2dArray.Load(int4(0));

    val += t3D[int3(0)];

    outputBuffer[0] = val.x;

    int w, h, l, lods;
    t2dArray.GetDimensions(0, w, h, l, lods);
    outputBuffer[1] = w + h + l + lods;
	return output;
}
