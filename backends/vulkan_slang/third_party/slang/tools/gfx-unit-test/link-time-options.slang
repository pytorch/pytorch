// link-time-options.slang

// Test that we can pass in additional compiler options
// at link time.

// Lowers to `DOWNSTREAM_VALUE`, a macro that we will define
// for downstream compilation through link-time options.
float getMacroDefinedForDownstream()
{
    __intrinsic_asm "(DOWNSTREAM_VALUE)";
}

[shader("compute")]
[numthreads(4,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> buffer)
{
    buffer[sv_dispatchThreadID.x] = getMacroDefinedForDownstream();
}
