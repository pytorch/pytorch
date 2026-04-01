//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage compute -emit-spirv-directly


struct Tester
{
    uint i;
};

struct Push
{
    Tester* ptr_array[2];
    uint * out_ptr;
};

[[vk::push_constant]] Push p;

// CHECK: OpEntryPoint

[shader("compute")]
[numthreads(1, 1, 1)]
void main(int id : SV_DispatchThreadID)
{
    uint i = p.ptr_array[0].i;
    *p.out_ptr = i;
}