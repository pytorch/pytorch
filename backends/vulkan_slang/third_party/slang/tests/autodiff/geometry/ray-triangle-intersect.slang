//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[Differentiable]
[PreferRecompute]
float3 diffRayIntersectTriangle(no_diff float3 rayOrigin, float3 rayDir, no_diff float3 p[3])
{
    float3 e1 = p[1] - p[0];
    float3 e2 = p[2] - p[0];
    float3 pVec = cross(rayDir, e2);
    float divisor = dot(pVec, e1);
    float3 s = rayOrigin - p[0];
    float u = dot(s, pVec) / divisor;
    float3 qVec = cross(s, e1);
    float v = dot(rayDir, qVec) / divisor;
    float t = dot(e2, qVec) / divisor;
    return float3(u, v, t);
}

[Differentiable]
[PreferRecompute]
float3 diffRayIntersectTriangle2(no_diff float3 rayOrigin, float3 rayTarget, no_diff float3 p[3])
{
    float3 rayDir = normalize(rayTarget - rayOrigin);
    float3 uvt = diffRayIntersectTriangle(rayOrigin, rayDir, p);
    float3 result = (1.f - uvt.x - uvt.y) * p[0] + uvt.x * p[1] + uvt.y * p[2];
    return result;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float3 shadePos = float3(0.674034, 0.0, 0.123171);
    float3 targetPos = float3(0.5, 0.2, -1.0);
    float3 triPos[3] = { float3(0.0, 1.0, -1.0), float3(1.0, 1.0, 0.0), float3(0.0, 1.0, 0.0) };

    // Forward-mode
    DifferentialPair<float3> dpIsectPos = fwd_diff(diffRayIntersectTriangle2)(
        shadePos,
        DifferentialPair<float3>(targetPos, float3(1.0, 0.0, 0.0)),
        triPos
    );

    outputBuffer[0] = dpIsectPos.d[0]; // Expect: 5.0
    outputBuffer[1] = dpIsectPos.d[1]; // Expect: 0.0
    outputBuffer[2] = dpIsectPos.d[2]; // Expect: 0.0

    // Reverse-mode
    DifferentialPair<float3> dpTargetPos = diffPair(targetPos, float3(0.f));
    bwd_diff(diffRayIntersectTriangle2)(
        shadePos,
        dpTargetPos,
        triPos,
        float3(1.f, 1.f, 1.f)
    );

    outputBuffer[3] = dpTargetPos.d[0]; // Expect: 5.0
    outputBuffer[4] = dpTargetPos.d[1]; // Expect: 32.4301
    outputBuffer[5] = dpTargetPos.d[2]; // Expect: 5.0
}
