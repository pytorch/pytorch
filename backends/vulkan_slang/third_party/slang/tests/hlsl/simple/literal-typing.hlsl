//TEST:COMPARE_HLSL: -profile cs_5_0 -entry main

// Confirm that we get the typing of literal suffixes correct

// A type created to cause type-checking failures downstream
struct Bad { int bad; };

// We define two overloads for `foo()`. The "right" one takes
// an unsigned integer, and returns it. The "wrong" one takes
// a signed integer and returns a `Bad`.

uint foo(uint x) { return x; }
Bad foo(int x) { Bad b; b.bad = x; return b; }

// The shader entry point will call `foo()` on a literal
// with a suffix that should make it unsigned, so that
// we either respect the suffix and call the right overload,
// or ignore it and call the wrong one.

#ifndef __SLANG__
#define b b_0
#endif

RWStructuredBuffer<uint> b;
[numthreads(32,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	b[tid.x] = foo(99u);
}