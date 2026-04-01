// trivial-copy.slang

// Copy the contents of "tex" into "buffer". These are for textures containing FLOAT data.
[shader("compute")]
[numthreads(4,1,1)]
void copyTexFloat4(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform RWTexture2D<float4> tex,
    uniform RWStructuredBuffer<float> buffer)
{
    float4 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 4] = result.r;
    buffer[sv_dispatchThreadID.x * 4 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 4 + 2] = result.b;
    buffer[sv_dispatchThreadID.x * 4 + 3] = result.a;
}
