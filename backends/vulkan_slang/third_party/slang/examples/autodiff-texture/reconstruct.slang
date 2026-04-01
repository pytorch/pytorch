// A compute shader to propagate gradients from high level mip(low-res) to lower level mip (high-res).

cbuffer Uniforms
{
    uint4 mipOffset[16];
    uint dstLayer;
    uint layerCount;
    uint width;
    uint height;
    RWStructuredBuffer<int> accumulateBuffer;
    RWStructuredBuffer<float> dstBuffer;
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
    var dstVal = int4(accumulateBuffer[dstOffset], accumulateBuffer[dstOffset + 1], accumulateBuffer[dstOffset + 2], accumulateBuffer[dstOffset + 3]);
    var newDstValToAdd = float3(0.0);
    if (dstVal.w > 0)
        newDstValToAdd = (float3)dstVal.xyz * float3(1.0 / (dstVal.w * 65536.0));

    float4 existingVal = 0.0;

    if (dstLayer < layerCount - 1)
    {
        uint parentOffset = mipOffset[(dstLayer + 1) / 4][(dstLayer + 1) % 4];
        uint parentW = dstW / 2;
        uint parentPixelLoc = parentOffset + ((y / 2) * parentW + (x / 2)) * 4;
        existingVal.x = dstBuffer[parentPixelLoc] * 0.25;
        existingVal.y = dstBuffer[parentPixelLoc + 1] * 0.25;
        existingVal.z = dstBuffer[parentPixelLoc + 2] * 0.25;
        existingVal.w = 0.0;
    }

    var newDstVal = existingVal + float4(newDstValToAdd, 0.0);
    dstBuffer[dstOffset] = newDstVal.x;
    dstBuffer[dstOffset + 1] = newDstVal.y;
    dstBuffer[dstOffset + 2] = newDstVal.z;
    dstBuffer[dstOffset + 3] = 1.0;
}
