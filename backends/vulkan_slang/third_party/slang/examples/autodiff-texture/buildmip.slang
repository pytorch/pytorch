// A compute shader to build a mip-map layer using box filtering.

cbuffer Uniforms
{
    uint dstWidth;
    uint dstHeight;
    RWTexture2D dstTexture;
    RWTexture2D srcTexture;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
    uint x = threadIdx.x;
    uint y = threadIdx.y;
    if (x >= dstWidth) return;
    if (y >= dstHeight) return;
    var val0 = srcTexture[uint2(x * 2, y * 2)];
    var val1 = srcTexture[uint2(x * 2 + 1, y * 2)];
    var val2 = srcTexture[uint2(x * 2, y * 2 + 1)];
    var val3 = srcTexture[uint2(x * 2 + 1, y * 2 + 1)];

    dstTexture[uint2(x, y)] = (val0 + val1 + val2 + val3) / 4;
}
