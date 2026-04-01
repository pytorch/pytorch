//TEST:SIMPLE(filecheck=CHECK): -profile lib_6_3 -entry main -stage compute -target spirv

[[vk::binding(0, 0)]]
Texture2DMS tex : register(t1);

RWStructuredBuffer<float4> outBuffer;

// CHECK: %{{.*}} = OpImageFetch %v4float %{{.*}} %{{.*}} Sample %int_0

[numthreads(4, 4, 1)]
void main(uint3 groupId : SV_GroupID)
{
    outBuffer[0] = tex.Load(int2(groupId.xy), 0);
}
