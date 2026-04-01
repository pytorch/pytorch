// used-parameters.slang

// Tests post-emit analysis of shader parameters to find out if they are used or not.

//TEST:REFLECTION:-stage compute -entry main -target hlsl


struct S
{
    uint2 Size;
};

ConstantBuffer<S> UsedCB;
ConstantBuffer<S> UnusedCB;

Texture2D UsedTexture;
Texture2D UnusedTexture;

Buffer<uint> UsedBuffer;
Buffer<uint> UnusedBuffer;

StructuredBuffer<uint> UsedStructuredBuffer;
StructuredBuffer<uint> UnusedStructuredBuffer;

RWTexture2D UsedRWTexture;
RWTexture2D UnusedRWTexture;

RWBuffer<uint> UsedRWBuffer;
RWBuffer<uint> UnusedRWBuffer;

RWStructuredBuffer<uint> UsedRWStructuredBuffer;
RWStructuredBuffer<uint> UnusedRWStructuredBuffer;

SamplerState UsedSampler;
SamplerState UnusedSampler;

uniform uint UsedUniform;
uniform uint UnusedUniform;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float A = UsedTexture[dispatchThreadID.xy].x;
    uint B = UsedBuffer[dispatchThreadID.x];
    uint C = UsedStructuredBuffer[dispatchThreadID.y];
    float D = UsedRWTexture[dispatchThreadID.xy].x;
    uint E = UsedRWBuffer[dispatchThreadID.y];
    float F = UsedTexture.SampleLevel(UsedSampler, float2(dispatchThreadID.xy) / float2(UsedCB.Size), 0).x;
    uint G = UsedUniform;

    UsedRWStructuredBuffer[dispatchThreadID.x + dispatchThreadID.y * UsedCB.Size.x] = uint(A) + B + C + uint(D) + E + uint(F) + G;
}