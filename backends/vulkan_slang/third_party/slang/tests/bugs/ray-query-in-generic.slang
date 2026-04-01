
//TEST(compute):COMPARE_COMPUTE:-d3d12 -output-using-type -use-dxil -profile sm_6_5 -render-feature ray-query
//TEST(compute):COMPARE_COMPUTE:-vk -output-using-type -render-feature ray-query

//TEST_INPUT: set scene = AccelerationStructure
uniform RaytracingAccelerationStructure scene;

struct Ray
{
    float3 origin;
    float tMin;
    float3 dir;
    float tMax;

    /**
     * Initializes a ray.
     */
    __init(float3 origin, float3 dir, float tMin = 0.f, float tMax = 10000.0)
    {
        this.origin = origin;
        this.dir = dir;
        this.tMin = tMin;
        this.tMax = tMax;
    }

    /**
     * Convert to DXR RayDesc.
     */
    RayDesc toRayDesc() { return { origin, tMin, dir, tMax }; }

    /**
     * Evaluate position on the ray.
     * @param[in] t Ray parameter.
     * @return Returns evaluated position.
     */
    float3 eval(float t) { return origin + t * dir; }
};

uint getCommittedStatus<let Flags : int>(RayQuery<Flags> q)
{
    return q.CommittedStatus();
}

bool traceSceneVisibilityRayImpl<let Flags : int>(const bool useAlphaTest, inout RayQuery<Flags> q, const Ray ray, uint rayFlags, uint instanceInclusionMask)
{
    var rayDesc = ray.toRayDesc();
    q.TraceRayInline(
        scene,
        rayFlags,
        0xff,
        rayDesc);
    q.Proceed();
    if (getCommittedStatus(q) == COMMITTED_TRIANGLE_HIT)
    {
        q.CommittedRayT();
        return true;
    }
    return false;
}

bool traceSceneVisibilityRay<let UseAlphaTest : int>(const Ray ray, uint rayFlags, uint instanceInclusionMask)
{
    RayQuery rayQuery;
    return traceSceneVisibilityRayImpl(false, rayQuery, ray, rayFlags, instanceInclusionMask);
}

//TEST_INPUT:set outputBuffer = out ubuffer(data=[0], stride=4)
RWStructuredBuffer<float> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(
    uint3 threadIdx: SV_DispatchThreadID)
{
    Ray ray = Ray(float3(0.1, 0.1, 0.0), float3(0, 0, 1), 0.1);
    if (traceSceneVisibilityRay<0>(ray, 0, 0xffff))
        outputBuffer[threadIdx.x] = 1.0;
}
