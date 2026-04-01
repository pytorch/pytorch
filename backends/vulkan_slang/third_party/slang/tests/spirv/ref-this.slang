//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: %[[PTR:[0-9a-zA-Z_]+]] = OpAccessChain %_ptr_PhysicalStorageBuffer_uint %{{.*}} %int_0
// CHECK: %{{.*}} = OpAtomicIAdd %uint %[[PTR]] %uint_1 %uint_0 %uint_1

struct Buf
{
    uint t;

    [__ref]
    func tester()
    {
        uint prev;
        InterlockedAdd(t, 1, prev);
    }
};

struct Push
{
    Buf * b;
};

[[vk::push_constant]] Push push;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    push.b->tester();
}