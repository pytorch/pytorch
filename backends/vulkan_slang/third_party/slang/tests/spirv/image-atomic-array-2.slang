//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: %{{.*}} = OpImageTexelPointer %{{.*}} %{{.*}} %{{.*}} %int_0
// CHECK: %{{.*}} = OpAtomicIAdd %ulong %{{.*}} %uint_1 %uint_0 %ulong_0

[[vk::binding(0,0)]] RWTexture2D<uint64_t> tex;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    InterlockedAdd(tex[uint2(0,0)], uint64_t(0));
}