// Test that the compiler can hoist the texture sample out of the loop.
// We currently have no way to verify the hoist actually take place, but this test
// allows us to manually inspect the generated code to see that happen.
// TODO: use a pattern checker on the resulting code to make sure there is only
// one call to `SampleLevel`.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE:-slang -compute -mtl -output-using-type -render-features argument-buffer-tier-2

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct Params : IDifferentiable
{
    Texture2D tex;
    SamplerState sampler;
}

//TEST_INPUT:set gParams = new Params{Texture2D(size=4, content=one), Sampler}
ParameterBlock<Params> gParams;

[BackwardDifferentiable]
float f(float x)
{
    float sum = 0.0;
    for (int i = 0; i < 3; i++)
    {
        float t = (no_diff gParams.tex.SampleLevel(gParams.sampler, float2(0.0), 0)).x;
        sum += t * x * x;
    }
    return sum;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    var dpa = diffPair(3.0);

    __bwd_diff(f)(dpa, 1.0);
    outputBuffer[0] = dpa.d.x; // Expect: 18
}
