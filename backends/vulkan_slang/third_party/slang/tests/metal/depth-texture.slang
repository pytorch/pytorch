//TEST:SIMPLE(filecheck=CHECK): -target metallib
//CHECK: sample_compare_depth_2d
Texture2D texture;
SamplerComparisonState sampler;

RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
void computeMain()
{
    output[0] = texture.SampleCmpLevelZero(sampler, float2(0, 0), 0);
}