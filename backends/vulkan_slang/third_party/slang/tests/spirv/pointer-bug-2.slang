//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -O0

struct Tester0
{
    uint i;
};

struct Tester1
{
    Tester0 * tester0;
};

struct Tester2
{
    Tester1* tester1;
};

struct Tester3
{
    uint* i_ptr;
};

struct Push
{
    Tester2 * tester2;
    Tester3 * tester3;
};

[[vk::push_constant]] Push push;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{    
    Tester3 tester3 = push.tester3[0];
    if (uint64_t(tester3.i_ptr) != 0)
    {
        return;
    }
}

// CHECK: OpEntryPoint