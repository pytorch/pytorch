RaytracingAccelerationStructure gScene_0 : register(t0);

RWStructuredBuffer<int > gOutput_0 : register(u0);

float3 helper_0(RayQuery<int(0) > q_0)
{
    RayQuery<int(0) > _S1 = q_0;

    RayDesc ray_0;
    float3 _S2 = (float3)0.0;

    ray_0.Origin = _S2;
    ray_0.Direction = _S2;
    ray_0.TMin = 0.0;
    ray_0.TMax = 1000.0;
    _S1.TraceRayInline(gScene_0, 0U, 4294967295U, ray_0);

    return _S1.WorldRayDirection();
}


[shader("compute")][numthreads(1, 1, 1)]
void computeMain(uint tid_0 : SV_DISPATCHTHREADID)
{
    RayQuery<int(0) > rayQuery_0;

    int _S3 = int(helper_0(rayQuery_0).x);

    gOutput_0[tid_0.x] = _S3;
    return;
}

