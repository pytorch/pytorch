//TEST:CROSS_COMPILE: -profile glsl_460+GL_NV_ray_tracing -entry HitMain -stage closesthit -target spirv-assembly
//TEST:SIMPLE(filecheck=DXIL): -target dxil -entry HitMain -stage closesthit -profile sm_6_5
//TEST:SIMPLE(filecheck=SPV): -target spirv

// DXIL: define void @
// SPV: OpEntryPoint

#define USE_RCP 0

struct ParameterBlockTest
{
    SamplerState sam;
    StructuredBuffer<float> sbuf;
};

ConstantBuffer<ParameterBlockTest> gParamBlock;

struct RayHitInfoPacked
{
    float4 PackedHitInfoA : PACKED_HIT_INFO_A;
};

[shader("closesthit")]
void HitMain(inout RayHitInfoPacked RayData, BuiltInTriangleIntersectionAttributes Attributes)
{
    float HitT = RayTCurrent();
    RayData.PackedHitInfoA.x = HitT;
    int offs = 0;
    uint use_rcp = USE_RCP;
    float offsfloat = gParamBlock.sbuf.Load(offs);

    use_rcp |= HitT > 0.0;

    if (use_rcp != 0)
        RayData.PackedHitInfoA.y = rcp(offsfloat);
    else if ((use_rcp > 0) & (offsfloat == 0.0))
        RayData.PackedHitInfoA.y = rsqrt(offsfloat + 1.0);
    else
        RayData.PackedHitInfoA.y = rsqrt(offsfloat);
}
