//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpImageSampleDrefExplicitLod
// CHECK: OpImageSampleDrefExplicitLod

Sampler2DShadow ss;

RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
void computeMain()
{
    output[0] = ss.SampleCmpLevelZero(float2(0.0), 0.5);

    float level = 1.5;
    output[1] = ss.SampleCmpLevel(float2(0.0), 0.5, level);
}
