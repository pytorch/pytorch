// texture.slang

// This class encapsulates a differentiable texture object that uses
// hardware sampling for the primal version, but substitutes a reference
// interpolation implementation to generate backward pass.
//
// This specific implementation also makes the choice to use fast fixed point
// atomics to accumulate the derivative (suitable for this example, but maybe
// not in general)
//
struct DifferentiableTexture
{
    RWStructuredBuffer<int> accumulateBuffer; // Per-mip-level accumulate buffer
    Texture2D texture;                        // Hardware texture handle.
    float minLOD;

    [BackwardDerivative(bwd_LoadTexel)]
    float4 LoadTexel(int3 location, constexpr int2 offset, uint dLayerW, uint dMipOffset)
    {
        return texture.Load(location, offset);
    }

    void bwd_LoadTexel(int3 location, constexpr int2 offset, uint dLayerW, uint dMipOffset, float4 val)
    {
        // Ignore alpha dimension for this example..
        int4 uval = int4(int3(val.xyz * 65536), 1);

        // We'll use fast fixed point atomics instead of floats.
        InterlockedAdd(accumulateBuffer[dMipOffset + ((uint)location.y * dLayerW + (uint)location.x) * 4 + 0], uval.x);
        InterlockedAdd(accumulateBuffer[dMipOffset + ((uint)location.y * dLayerW + (uint)location.x) * 4 + 1], uval.y);
        InterlockedAdd(accumulateBuffer[dMipOffset + ((uint)location.y * dLayerW + (uint)location.x) * 4 + 2], uval.z);
        InterlockedAdd(accumulateBuffer[dMipOffset + ((uint)location.y * dLayerW + (uint)location.x) * 4 + 3], uval.w);
    }

    // Software reference implementation of linear filtering.
    [BackwardDifferentiable]
    float4 sampleTexture_linear(uint lod, float2 uv, uint w, uint h)
    {
        w >>= lod;
        h >>= lod;
        uv = uv - no_diff(floor(uv));
        float2 loc = uv * float2(w, h) - float2(0.5);
        float x0 = no_diff(floor(loc.x));
        float y0 = no_diff(floor(loc.y));
        float fracX = loc.x - x0;
        float fracY = loc.y - y0;
        float x1 = x0 + 1;
        float y1 = y0 + 1;
        if (x0 < 0) x0 += w;
        if (y0 < 0) y0 += h;
        if (x1 >= w) x1 -= w;
        if (y1 >= h) y1 -= h;
        float weight0 = 1.0f - fracY;
        float weight1 = fracY;
        float weight00 = weight0 * (1.0f - fracX);
        float weight01 = weight0 * fracX;
        float weight10 = weight1 * (1.0f - fracX);
        float weight11 = weight1 * fracX;

        uint dLayerW = w >>= lod;
        var offset = mipOffset[lod / 4][lod % 4];
        return LoadTexel(int3(int(x0), int(y0), int(lod)), int2(0), dLayerW, offset) * weight00 +
               LoadTexel(int3(int(x1), int(y0), int(lod)), int2(0), dLayerW, offset) * weight01 +
               LoadTexel(int3(int(x0), int(y1), int(lod)), int2(0), dLayerW, offset) * weight10 +
               LoadTexel(int3(int(x1), int(y1), int(lod)), int2(0), dLayerW, offset) * weight11;
    }

    // Software reference implementation of trilinear filtering.
    [BackwardDifferentiable]
    float4 sampleTexture_trilinear(uint w, uint h, uint levels, float2 uv, float2 dX, float2 dY)
    {
        dX = dX * float2(w, h);
        dY = dY * float2(w, h);

        // Isotropic filter.
        float lengthX = length(dX);
        float lengthY = length(dY);
        float LOD = log2(max(lengthX, lengthY));
        float maxLOD = levels - 1;
        float clampedLOD = max(minLOD, (min(maxLOD, LOD)));

        float lodFrac = clampedLOD - no_diff(floor(clampedLOD));
        uint lod0 = (uint)floor(clampedLOD);
        uint lod1 = min(levels - 1, lod0 + 1);
        float weightLod0 = 1.0 - lodFrac;
        float weightLod1 = lodFrac;

        let v0 = sampleTexture_linear(lod0, uv, w, h) * weightLod0;
        let v1 = sampleTexture_linear(lod1, uv, w, h) * weightLod1;
        return v0 + v1;
    }

    // Note that there is no need to mark this [BackwardDifferentiable] since it has a substitute
    // that is marked [BackwardDifferentiable]. The compiler automatically considers a call to
    // sample() to be differentiable.
    //
    static float4 sample(DifferentiableTexture t, SamplerState s, float2 uv, float2 dX, float2 dY)
    {
        return t.texture.Sample(s, uv);
    }

    // Software reference implementation of DifferentiableTexture.sample (trilinear only in this example)
    [PrimalSubstituteOf(DifferentiableTexture.sample)]
    [BackwardDifferentiable]
    static float4 sample_reference_impl(DifferentiableTexture t, SamplerState s, float2 uv, float2 dX, float2 dY)
    {
        uint w;
        uint h;
        uint levels;
        t.texture.GetDimensions(0, w, h, levels);
        return t.sampleTexture_trilinear(w, h, levels, uv, dX, dY);
    }
}

cbuffer Uniforms
{
    float4x4 modelViewProjection;
    uint4 mipOffset[16];

    Texture2D texRef;
    SamplerState sampler;
    DifferentiableTexture bwdTexture;
}

struct AssembledVertex
{
    float3	position : POSITION;
};

struct Fragment
{
    float4 color;
};

struct VertexStageOutput
{
    float2 uv : UV;
    float4          sv_position     : SV_Position;
};

[BackwardDifferentiable]
float4 shadeFragment(float2 uv)
{
    uv = uv * 2;

    // Compute fragment differentials using shader intrinsics.
    float2 dX = no_diff ddx_coarse(uv);
    float2 dY = no_diff ddy_coarse(uv);

    float3 color = DifferentiableTexture.sample(bwdTexture, sampler, uv, dX, dY).xyz;
    return float4(color, 1.0);
}

[BackwardDifferentiable]
float3 loss(no_diff float2 uv, no_diff float4 screenPos)
{
    float3 refColor = (no_diff texRef.Load(int3(int2(screenPos.xy), 0))).xyz;
    float3 rs = shadeFragment(uv).xyz - refColor;
    rs *= rs;
    return rs;
}

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;

    output.uv = position.xy;
    output.sv_position = mul(modelViewProjection, float4(position, 1.0));

    return output;
}

float3 sqr(float3 v) { return v * v; }

[shader("fragment")]
float4 fragmentMain(
    float2 uv : UV) : SV_Target
{
    return shadeFragment(uv);
}

[shader("fragment")]
float4 diffFragmentMain(
    float2 uv : UV,
    float4 screenPos : SV_POSITION) : SV_Target
{
    __bwd_diff(loss)(uv, screenPos, float3(1.0));
    return float4(loss(uv, screenPos), 1.0);
}
