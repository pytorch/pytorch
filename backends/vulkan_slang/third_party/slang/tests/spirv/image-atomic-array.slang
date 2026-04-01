//TEST:SIMPLE(filecheck=CHECK): -target spirv

import glsl;
[[vk::binding(0,0)]] RWTexture2D<uint64_t> tex[];

// CHECK: OpEntryPoint
// CHECK: %{{.*}} = OpImageTexelPointer %{{.*}} %{{.*}} %{{.*}} %int_0
// CHECK: %{{.*}} = OpAtomicIAdd %ulong %{{.*}} %uint_1 %uint_0 %ulong_0
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    imageAtomicAdd(tex[0], int2(0,0), uint64_t(0));
}