//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

// CHECK: OpEntryPoint

struct Tester
{
    uint values[2];
};

struct TestPush
{
    Tester* src;
    uint* dst;
};

[[vk::push_constant]] TestPush test_p;

[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    uint value = test_p.src.values[0];
    *test_p.dst = value;
}
