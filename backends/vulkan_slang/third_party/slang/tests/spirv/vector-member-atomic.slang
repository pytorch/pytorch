//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: %[[PTR:[0-9a-zA-Z_]+]] = OpAccessChain %_ptr_PhysicalStorageBuffer_uint %16 %int_0
// CHECK: %{{.*}} = OpAtomicIAdd %uint %[[PTR]] %uint_1 %uint_0 %uint_1


struct Push2
{
    uint4 * value;
};

[[vk::push_constant]] Push2 push2;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    InterlockedAdd(push2.value[0].x, 1);
}