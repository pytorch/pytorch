//TEST:COMPARE_HLSL: -profile cs_5_0 -entry main

// Confirm that we properly pass along the `numthreads` attribute on an entry point.

#ifndef __SLANG__
#define b b_0
#endif

RWStructuredBuffer<float> b;

[numthreads(32,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	b[tid.x] = b[tid.x + 1] + 1.0f;
}