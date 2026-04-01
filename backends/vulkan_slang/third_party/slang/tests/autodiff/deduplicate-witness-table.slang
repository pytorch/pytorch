//TEST:SIMPLE(filecheck=CHK):-stage compute -entry computeMain -target hlsl

//CHK: struct DiffPair_1
//CHK-NOT: struct DiffPair_2

RWTexture2D<float> gOutputColor;

struct ShadingFrame : IDifferentiable
{
    float3 T;
}

[Differentiable]
float computeRay()
{
    float3 dir = 1.f;
    return dot(dir, dir);
}

[Differentiable]
float paramRay()
{
    DifferentialPair<float> dpDir = fwd_diff(computeRay)();
    return dpDir.p;
}

[Shader("compute")]
[NumThreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    DifferentialPair<float> dpColor = fwd_diff(paramRay)();
    gOutputColor[0] = dpColor.p;
}
