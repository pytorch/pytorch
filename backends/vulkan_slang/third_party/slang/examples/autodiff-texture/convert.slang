// A compute shader to convert from a buffer into a mip-map texture.
cbuffer Uniforms
{
    uint4 mipOffset[16];
    uint dstLayer;
    uint width;
    uint height;
    RWStructuredBuffer<float> srcBuffer;
    RWTexture2D dstTexture;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
    uint x = threadIdx.x;
    uint y = threadIdx.y;
    uint dstW = width >> dstLayer;
    uint dstH = height >> dstLayer;
    if (x >= dstW) return;
    if (y >= dstH) return;
    uint dstOffset = mipOffset[dstLayer / 4][dstLayer % 4] + (y * dstW + x) * 4;
    var dstVal = float4(srcBuffer[dstOffset], srcBuffer[dstOffset + 1], srcBuffer[dstOffset + 2], srcBuffer[dstOffset + 3]);
    dstTexture[uint2(x, y)] = dstVal;
}
