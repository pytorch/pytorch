// compute-trivial.slang - Simple shader that takes a buffer and increments all elements by 1.

uniform RWStructuredBuffer<float> buffer;

[shader("compute")]
[numthreads(4,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    var input = buffer[sv_dispatchThreadID.x];
    buffer[sv_dispatchThreadID.x] = input + 1.0f;
}
