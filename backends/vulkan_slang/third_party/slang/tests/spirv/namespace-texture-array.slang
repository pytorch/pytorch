//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly
struct ComputePush
{
    uint image_id;
};
[[vk::push_constant]] ComputePush p;

namespace test_namespace
{
    [[vk::binding(0, 0)]] RWTexture2D<float4> textureTable[];
}

// CHECK: OpEntryPoint

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 pixel_i : SV_DispatchThreadID)
{
    test_namespace.textureTable[p.image_id][pixel_i.xy] = float4(0,1,0,0);
}