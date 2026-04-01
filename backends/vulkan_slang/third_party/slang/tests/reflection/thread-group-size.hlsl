//TEST:REFLECTION:-profile cs_5_0 -target hlsl -no-codegen

// Confirm that we provide reflection data for the `numthreads` attribute

RWStructuredBuffer<float> b;

[numthreads(3,5,7)]
void main(uint3 tid : SV_DispatchThreadID)
{
	b[tid.x] = b[tid.x + 1] + 1.0f;
}