// sampler-array.slang

// Test sampler array parameters.

struct S1
{
    Texture2D tex[32];
    SamplerState samplers[32];
    float data;
    float test(int i)
    {
        return tex[i].SampleLevel(samplers[i], float2(0.0, 0.0), 0.0).x + data;
    }
}

struct S0
{
    float data;
    RaytracingAccelerationStructure acc;
    ParameterBlock<S1> s;
}

ParameterBlock<S0> g;
RWStructuredBuffer<float> buffer;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    buffer[0] = g.data * g.s.test(sv_dispatchThreadID.x);
}
