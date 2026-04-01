//TEST:SIMPLE(filecheck=WGSL): -stage vertex -entry main -target wgsl

static const int cubeCount = 3;

struct PerInstanceUniforms
{
    float4x4 model;
    float4x4 normalModel;
};

// WGSL: @binding(0) @group(0) var<uniform> perInstance_0 : array<PerInstanceUniforms_std140_0, i32(3)>;
[[vk::binding(0, 0)]] ConstantBuffer<PerInstanceUniforms> perInstance[cubeCount] : register(b0, space0);

struct VertexInput
{
    float4 position : POSITION0;
};

float4 main(VertexInput input)
{
    // Only here to ensure that perInstance is not ignored.
    float4 pos = mul(input.position, perInstance[0].model);
    return pos;
}
