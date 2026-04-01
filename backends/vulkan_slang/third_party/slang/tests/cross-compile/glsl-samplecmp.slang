Texture2D shadowMap;
SamplerComparisonState sampler;
float4 main(float4 p : SV_POSITION)
{
    return shadowMap.SampleCmp(sampler, float2(0.0), 0.0);
}