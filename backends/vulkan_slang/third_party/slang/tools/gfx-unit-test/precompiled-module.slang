import "precompiled-module-imported";

using namespace ns;

// Main entry-point. 

[shader("compute")]
[numthreads(4, 1, 1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer <float> buffer)
{
    buffer[sv_dispatchThreadID.x] = helperFunc() + helperFunc1();
}
