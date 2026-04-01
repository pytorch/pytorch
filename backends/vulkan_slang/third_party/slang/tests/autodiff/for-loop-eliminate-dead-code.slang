//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

static const uint kMaxSurfaceBounces = 2;

[BackwardDifferentiable]
void updatePathContrib(inout float3 result, float3 val)
{
    result = result + val;
}

[BackwardDifferentiable]
float3 evalPath(uint2 pixel)
{
    float3 result = float3(0.f);
    float3 thp = float3(1.f);

    [MaxIters(kMaxSurfaceBounces)]
    for (int i = 0; i < kMaxSurfaceBounces; i++)
    {
        float3 neeContrib = float3(1.f);
        updatePathContrib(result, thp * neeContrib);
        thp = thp * float3(0.9f);
    }
    return result;
}

void execute(const uint2 pixel)
{
    float3 dColor = float3(1.f);
    __bwd_diff(evalPath)(pixel, dColor);
}

[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    execute(dispatchThreadID.xy);
}
