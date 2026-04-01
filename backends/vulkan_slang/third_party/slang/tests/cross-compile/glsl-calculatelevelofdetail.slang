//TEST:CROSS_COMPILE(filecheck=CHECK): -profile ps_5_0 -entry main -target glsl

// CHECK: textureQueryLod(sampler2D(
// CHECK: textureQueryLod(sampler2D(
// CHECK: textureQueryLod(sampler2DShadow(
// CHECK: textureQueryLod(sampler2DShadow(

Texture2D t;
SamplerState s;
SamplerComparisonState sc;

float main()
{
    float result = 0.0;
    result += t.CalculateLevelOfDetail(s, float2(0,0)) + t.CalculateLevelOfDetailUnclamped(s, float2(0,0));
    result += t.CalculateLevelOfDetail(sc, float2(0,0)) + t.CalculateLevelOfDetailUnclamped(sc, float2(0,0));
    return result;
}
