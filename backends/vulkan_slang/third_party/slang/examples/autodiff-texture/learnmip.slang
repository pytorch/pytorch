// A compute shader to add gradients to a mip-map texture.

cbuffer Uniforms
{
    uint dstWidth;
    uint dstHeight;
    float learningRate;
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
    var val = srcTexture[uint2(x, y)];
    dstTexture[uint2(x, y)] = float4((dstTexture[uint2(x, y)] - val * learningRate).xyz, 1.0);
}
