//TEST:SIMPLE(filecheck=CHECK_HLSL): -stage fragment -entry fragmentMain -target hlsl
//TEST:SIMPLE(filecheck=CHECK_SPIRV): -stage fragment -entry fragmentMain -target spirv

Texture2D t;
SamplerState s;
SamplerComparisonState sc;

float fragmentMain()
{
    float result = 0.0;

    // CHECK_SPIRV: OpCapability ImageQuery

    // CHECK_HLSL: CalculateLevelOfDetail
    // CHECK_HLSL: CalculateLevelOfDetailUnclamped
    // CHECK_SPIRV: OpImageQueryLod
    // CHECK_SPIRV: OpImageQueryLod
    result += t.CalculateLevelOfDetail(s, float2(0,0)) + t.CalculateLevelOfDetailUnclamped(s, float2(0,0));

    // CHECK_HLSL: CalculateLevelOfDetail
    // CHECK_HLSL: CalculateLevelOfDetailUnclamped
    // CHECK_SPIRV: OpImageQueryLod
    // CHECK_SPIRV: OpImageQueryLod
    result += t.CalculateLevelOfDetail(sc, float2(0,0)) + t.CalculateLevelOfDetailUnclamped(sc, float2(0,0));

    return result;
}
