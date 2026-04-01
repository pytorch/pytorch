//TEST(compute):COMPARE_COMPUTE:-d3d12 -output-using-type -use-dxil -profile sm_6_5 -render-feature ray-query
//TEST(compute):COMPARE_COMPUTE:-vk -output-using-type -render-feature ray-query

//TEST_INPUT: set scene = AccelerationStructure
uniform RaytracingAccelerationStructure scene;

//TEST_INPUT:set outputBuffer = out ubuffer(data=[0], stride=4)
RWStructuredBuffer<float> outputBuffer;

bool traceRayClosestHit(
    float3 rayOrigin,
    float3 rayDir,
    out float t)
{
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.TMin = 0.01f;
    ray.Direction = rayDir;
    ray.TMax = 1e4f;
    RayQuery<RAY_FLAG_NONE> q;
    let rayFlags = RAY_FLAG_NONE;

    q.TraceRayInline(
        scene,
        rayFlags,
        0xff,
        ray);

    q.Proceed();
    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        t = q.CommittedRayT();
        //primitiveIndex = q.CommittedPrimitiveIndex();
        return true;
    }
    unused(t);
    return false;
}

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(
    uint3 threadIdx : SV_DispatchThreadID)
{
    float t = 0.0;
    traceRayClosestHit(float3(0.1, 0.1, 0.0), float3(0,0,1), t);
    outputBuffer[threadIdx.x] = t;
}