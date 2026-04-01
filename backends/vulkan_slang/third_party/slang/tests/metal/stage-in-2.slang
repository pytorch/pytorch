//TEST:SIMPLE(filecheck=METAL): -target metal
//TEST:SIMPLE(filecheck=METAL_ASM): -target metallib
//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage fragment -entry fragmentMain
//TEST:SIMPLE(filecheck=WGSLSPIRV): -target wgsl-spirv-asm

// METAL_ASM: define {{.*}} @vertexMain
// METAL_ASM: define {{.*}} @fragmentMain

// Check that we don't flatten stage-input parameters that have user semantics.

// METAL: struct pixelInput
// METAL-NEXT: {
// METAL-NEXT: {{\[\[}}user(COARSEVERTEX){{\]\]}};

//WGSLSPIRV: %vertexMain = OpFunction %
//WGSLSPIRV: %fragmentMain = OpFunction %

//WGSL:     @location(0) output
//WGSL:     @location(0) color

// Uniform data to be passed from application -> shader.
cbuffer Uniforms
{
    float4x4 modelViewProjection;
}

// Per-vertex attributes to be assembled from bound vertex buffers.
struct AssembledVertex
{
    float3	position : POSITION;
    float3	color    : COLOR;
};

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 color;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};

// Vertex  Shader

struct VertexStageOutput
{
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;
    float3 color    = assembledVertex.color;

    output.coarseVertex.color = color;
    output.sv_position = mul(modelViewProjection, float4(position, 1.0));

    return output;
}

// Fragment Shader

[shader("fragment")]
float4 fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex) : SV_Target
{
    float3 color = coarseVertex.color;

    return float4(color, 1.0);
}
