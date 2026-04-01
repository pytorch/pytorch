// tests/compute/*type-param* tests are broken #5627
//DISABLED_TEST(compute):COMPARE_RENDER_COMPUTE:

//TEST_INPUT: global_type AssembledVertex
//TEST_INPUT: ubuffer(data=[0], stride=4):out

// Testing associated types in a varying parameter field

interface IColorSet
{
    float3 getColor();
}

struct NoColorSet : IColorSet
{
    float3 getColor() { return float3(0.0); }
};

struct SingleColorSet : IColorSet
{
    float3 color;
    float3 getColor() { return color; }
};

interface IVertexFormat
{
    associatedtype ColorSet : IColorSet;
    ColorSet getColorSet();
    float3 getPosition();
    float2 getUv();
}

type_param TVertex : IVertexFormat;

cbuffer Uniforms
{
    float4x4 modelViewProjection;
}
RWStructuredBuffer<float> outputBuffer;

struct AssembledVertex : IVertexFormat
{
    typedef SingleColorSet ColorSet;
    float3	position;
    SingleColorSet	colorSet;
    float2  uv;
    ColorSet getColorSet() { return colorSet; }
    float3 getPosition() { return position; }
    float2 getUv() { return uv; }
};

struct CoarseVertex
{
    TVertex.ColorSet	color;
    float2  uv;
};

struct Fragment
{
    float4 color;
};


// Vertex  Shader

struct VertexStageInput
{
    TVertex assembledVertex	: A;
};

struct VertexStageOutput
{
    CoarseVertex	coarseVertex	: CoarseVertex;
    float4			sv_position		: SV_Position;
};

VertexStageOutput vertexMain(VertexStageInput input)
{
    VertexStageOutput output;

    float3 position = input.assembledVertex.getPosition();
    output.sv_position = mul(modelViewProjection, float4(position, 1.0));
    output.coarseVertex.uv = input.assembledVertex.getUv();
    output.coarseVertex.color = input.assembledVertex.getColorSet();
    return output;
}

// Fragment Shader

struct FragmentStageInput
{
    CoarseVertex coarseVertex : CoarseVertex;
};

struct FragmentStageOutput
{
    Fragment fragment : SV_Target;
};

FragmentStageOutput fragmentMain(FragmentStageInput input)
{
    FragmentStageOutput output;
    float3 color = input.coarseVertex.color.getColor();
    float2 uv = input.coarseVertex.uv;
    output.fragment.color = float4(color, 1.0);
    outputBuffer[0] = 1.0;
    return output;
}
