// buffer-layout.slang

// The goal of this test is to make sure that constant and structured
// buffers obey the rules we expect of the each target API.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -shaderobj

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct A
{
    float x;
    float y;
}

struct S
{
    // The first field in a struct isn't going to be that
    // interesting, because it will always get offset zero,
    // so we just use this to establish a poorly-aligned
    // starting point for the next field.
    //
    //          offset  size    alignment
    //
    //          0       4       4
    //
    float z;

    // The `std140` and D3D constant buffer ruless both
    // ensure a minimum of 16-byte alignment on `struct`
    // types, but differ in that D3D does not round up
    // the total size of a type to its alignment.
    //
    // The `std430` and structured buffer rules don't
    // perform any over-alignment on `struct` types and
    // instead align them using the "natural" rules one
    // might expect of, e.g., a C compiler.
    //
    //          offset  size    alignment
    //
    // cbuffer  16      8       16
    // std140   16      16      16
    //
    // struct   4       8       4
    // std430   4       8       4
    //
    A      a;

    // Now we insert an ordinary `int` field just as
    // a way to probe the offset so far.
    //
    //          offset  size    alignment
    //
    // cbuffer  24      4       4
    // std140   32      4       4
    //
    // struct   12      4       4
    // std430   12      4       4
    //
    int    b;

    // As our next stress-test case, we will insert an
    // array with elements that aren't a multiple of
    // 16 bytes in size.
    //
    // The contant/uniform buffer rules will set the
    // array stride to a multiple of 16 bytes in this case.
    // The only difference between D3D rules and `std140`
    // here is that D3D does not round up the size to
    // the alignment.
    //
    // The structured/std430 rules don't do anything
    // to over-align an array, so it is laid out relatively
    // naturally, but note that D3D still follows its rule
    // of not letting a vector "straddle" a 16-byte boundary,
    // even if it doesn't bump up the alignment of
    // vector types.
    //
    //          offset  size    alignment
    //
    // cbuffer  32      24      16
    // std140   48      32      32
    //
    // struct   16      16      4
    // std430   16      16      8
    //
    float2 c[2];

    // Now we put in one more ordinary `int` field
    // just to probe the offset computed so far.
    //          offset  size    alignment
    //
    // cbuffer  56      4      4
    // std140   80      4      4
    //
    // struct   32      4      4
    // std430   32      4      4
    //
    int    d;
}

//TEST_INPUT:cbuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32]):name=cb
ConstantBuffer<S> cb;

//TEST_INPUT:ubuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32],stride=4):name=sb
RWStructuredBuffer<S> sb;

int test(int val)
{
    val = val+1;
    val = val*16  + cb.b;
    val = val*256 + cb.d;
    val = val*256 + sb[0].b;
    val = val*256 + sb[0].d;
    return val;
}

[numthreads(4, 1, 1)]
void computeMain(
    int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;

    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
