//Tests automatic synthesis of Differential type requirement.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type -dx12
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct PathData : IDifferentiable
{
    float3 thp;    
    uint length;
    bool terminated;
    bool isHit;

    [BackwardDifferentiable]
    __init()
    {
        this.thp = float3(1.f);
        this.length = 0;
        this.terminated = false;
        this.isHit = false;
    }
}

bool traceRayInline(uint length)
{
    if (length < 2) return true;
    else return false;
}

float3 getAlbedo(uint length)
{
    return float3(0.9f, 1.f, 1.f);
}

float3 getAlbedoDerivative(uint length)
{
    return float3(1.f, 0.f, 0.f);
}

[ForwardDerivativeOf(getAlbedo)]
[TreatAsDifferentiable]
DifferentialPair<float3> __fwd_d_getAlbedo(uint length)
{
    float3 primalValue = getAlbedo(length);
    float3 derivativeValue = no_diff getAlbedoDerivative(length);
    return DifferentialPair<float3>(primalValue, derivativeValue);
}

[BackwardDerivativeOf(getAlbedo)]
[TreatAsDifferentiable]
void __bwd_d_getAlbedo(uint length, float3 dOut)
{
    outputBuffer[2] += dOut.x;
}

[BackwardDifferentiable]
void handleHit(inout PathData pathData)
{
    if (pathData.length >= 2)
    {
        pathData.terminated = true;
        return;
    }

    float3 albedo = getAlbedo(pathData.length);
    pathData.thp *= albedo;
    pathData.length++;
}

[BackwardDifferentiable]
[PreferRecompute]
float3 tracePath()
{
    PathData pathData = PathData();

    if (traceRayInline(pathData.length))
    {
        pathData.isHit = true;
    }
    else
    {
        pathData.terminated = true;
        pathData.isHit = false;
    }

    [MaxIters(4)]
    while (!pathData.terminated)
    {
        if (pathData.isHit)
        {
            handleHit(pathData);

            //pathData.isHit = traceRayInline(pathData.length);
            if (!traceRayInline(pathData.length)) pathData.isHit = false;
            else pathData.isHit = true;
        }
        else
        {
            pathData.terminated = true;
        }
    }
    return pathData.thp;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispathThreadID: SV_DispatchThreadID)
{
    DifferentialPair<float3> dpThp = __fwd_diff(tracePath)();
    outputBuffer[0] = dpThp.p.x;
    outputBuffer[1] = dpThp.d.x;

    __bwd_diff(tracePath)(float3(1.f, 0.f, 0.f));
}
