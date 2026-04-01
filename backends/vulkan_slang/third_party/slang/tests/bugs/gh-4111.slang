//TEST:SIMPLE(filecheck=CHECK): -target spirv

// Check that a pointer typed local variable has `AliasedPointer` decoration.

// CHECK: OpDecorate %b AliasedPointer
struct Buf
{
    uint test;
};

struct Push
{
    bool a_or_b;
    Buf * a;
    Buf * b;
};

[[vk::push_constant]] Push push;

func tester(Buf * buf)
{
    buf->test = 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    Buf * b = push.a_or_b ? push.a : push.b;
    tester(b);
}