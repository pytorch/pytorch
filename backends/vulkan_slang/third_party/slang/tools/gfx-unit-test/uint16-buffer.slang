// uint16-buffer.slang - Simple shader that takes a buffer of uint16 type and increments all elements by 1.

// This is to verify that GFX can correct set correct buffer strides for structured buffer bindings.

uniform RWStructuredBuffer<uint16_t> buffer;

[shader("compute")]
[numthreads(4,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    var input = buffer[sv_dispatchThreadID.x];
    buffer[sv_dispatchThreadID.x] = input + 1;
}
