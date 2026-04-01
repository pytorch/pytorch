// struct-bit-cast.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that bit_cast works for bit-reinterpreting one struct type as another.

struct Foo
{
    uint a;
    float b;
    float2 fvec;
}

struct Inner
{
    int v;
    uint s;
}

struct Bar
{
    int u;
    Inner i;
    uint t;
}

int test0(int val)
{
    Bar b;
    b.u = val;
    b.i.v = asint(2.0f);
    b.i.s = asuint(1.25);
    b.t = asuint(0.25);
    Foo f = bit_cast<Foo, Bar>(b);
    return int(f.a) + (int)f.b + int(float(f.fvec.x / f.fvec.y)); // val + 2 + 5
}

struct Smaller
{
    int s;
}

struct Larger
{
    int x, y;
}

int test1()
{
    return 1;
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test0(inVal) + test1();
    outputBuffer[tid] = outVal;
}
