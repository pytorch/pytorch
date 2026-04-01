// format-test-shaders.slang - Shaders used by the unit tests in format-uint-tests.cpp

// Copy the contents of "tex" into "buffer". These are for textures containing UINT data.
[shader("compute")]
[numthreads(4,1,1)]
void copyTexUint4(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<uint4> tex,
    uniform RWStructuredBuffer<uint> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    uint4 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 4] = result.r;
    buffer[sv_dispatchThreadID.x * 4 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 4 + 2] = result.b;
    buffer[sv_dispatchThreadID.x * 4 + 3] = result.a;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexUint3(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<uint4> tex,
    uniform RWStructuredBuffer<uint> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    uint4 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 3] = result.r;
    buffer[sv_dispatchThreadID.x * 3 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 3 + 2] = result.b;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexUint2(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<uint2> tex,
    uniform RWStructuredBuffer<uint> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    uint2 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 2] = result.r;
    buffer[sv_dispatchThreadID.x * 2 + 1] = result.g;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexUint(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<uint> tex,
    uniform RWStructuredBuffer<uint> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    buffer[sv_dispatchThreadID.x] = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
}

// Copy the contents of "tex" into "buffer". These are for textures containing SINT data.
[shader("compute")]
[numthreads(4,1,1)]
void copyTexInt4(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<int4> tex,
    uniform RWStructuredBuffer<int> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    int4 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 4] = result.r;
    buffer[sv_dispatchThreadID.x * 4 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 4 + 2] = result.b;
    buffer[sv_dispatchThreadID.x * 4 + 3] = result.a;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexInt3(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<int4> tex,
    uniform RWStructuredBuffer<int> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    int4 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 3] = result.r;
    buffer[sv_dispatchThreadID.x * 3 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 3 + 2] = result.b;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexInt2(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<int2> tex,
    uniform RWStructuredBuffer<int> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    int2 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 2] = result.r;
    buffer[sv_dispatchThreadID.x * 2 + 1] = result.g;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexInt(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<int> tex,
    uniform RWStructuredBuffer<int> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    buffer[sv_dispatchThreadID.x] = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
}

// Copy the contents of "tex" into "buffer". These are for textures containing FLOAT data.
[shader("compute")]
[numthreads(4,1,1)]
void copyTexFloat4(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<float4> tex,
    uniform RWStructuredBuffer<float> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    float4 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 4] = result.r;
    buffer[sv_dispatchThreadID.x * 4 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 4 + 2] = result.b;
    buffer[sv_dispatchThreadID.x * 4 + 3] = result.a;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexFloat3(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<float4> tex,
    uniform RWStructuredBuffer<float> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    float4 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 3] = result.r;
    buffer[sv_dispatchThreadID.x * 3 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 3 + 2] = result.b;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexFloat2(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<float2> tex,
    uniform RWStructuredBuffer<float> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    float2 result = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
    buffer[sv_dispatchThreadID.x * 2] = result.r;
    buffer[sv_dispatchThreadID.x * 2 + 1] = result.g;
}

[shader("compute")]
[numthreads(4,1,1)]
void copyTexFloat(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<float> tex,
    uniform RWStructuredBuffer<float> buffer)
{
    uint width;
    uint height;
    tex.GetDimensions(width, height);
    buffer[sv_dispatchThreadID.x] = tex[uint2(sv_dispatchThreadID.x % width, sv_dispatchThreadID.x / width)];
}

// Sample from "tex" at texture coordinates (0.5, 0.5) and save the results in "buffer".
[shader("compute")]
[numthreads(4,1,1)]
void sampleTex(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<float4> tex,
    uniform SamplerState sampler,
    uniform RWStructuredBuffer<float> buffer)
{
    float4 result = tex.SampleLevel(sampler, float2(0.5, 0.5), 0);
    buffer[sv_dispatchThreadID.x * 4] = result.r;
    buffer[sv_dispatchThreadID.x * 4 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 4 + 2] = result.b;
    buffer[sv_dispatchThreadID.x * 4 + 3] = result.a;
}

// Sample from "tex" at texture coordinates (0.5, 0.5) and save the resuls in "buffer".
// This should only be used with textures containing two mip levels.
[shader("compute")]
[numthreads(2,1,1)]
void sampleMips(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D<float4> tex,
    uniform SamplerState sampler,
    uniform RWStructuredBuffer<float> buffer)
{
    float4 result = tex.SampleLevel(sampler, float2(0.5, 0.5), float(sv_dispatchThreadID.x));
    buffer[sv_dispatchThreadID.x * 4] = result.r;
    buffer[sv_dispatchThreadID.x * 4 + 1] = result.g;
    buffer[sv_dispatchThreadID.x * 4 + 2] = result.b;
    buffer[sv_dispatchThreadID.x * 4 + 3] = result.a;
}
