// buffer-layout.slang

// This test mirrors `tests/compute/buffer-layout.slang`, and it meant
// to confirm that our reflection logic correctly reports the offsets
// that the compute test sees in practice.

//TEST:REFLECTION:-stage compute -entry main -target hlsl -no-codegen
//TEST:REFLECTION:-stage compute -entry main -target spirv -no-codegen

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

ConstantBuffer<S> cb;
RWStructuredBuffer<S> sb;

[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
}