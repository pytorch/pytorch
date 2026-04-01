// ray-query-subroutine.slang

// Regression test for passing a `RayQuery` value into a subroutine.

//TEST:CROSS_COMPILE: -profile sm_6_5 -stage compute -entry computeMain -target dxil-assembly

RWStructuredBuffer<int> gOutput;
RaytracingAccelerationStructure gScene;

float3 helper<let N : uint>(inout RayQuery<N> q)
{
    RayDesc ray;
    ray.Origin = 0;
    ray.Direction = 0;
    ray.TMin = 0;
    ray.TMax = 1000.0;
    q.TraceRayInline(
        /* accellerationStructure: */   gScene,
        /* rayFlags: */                 N,
        /* instanceInclusionmask: */    0xFFFFFFFF,
        /* ray: */                      ray );

    return q.WorldRayDirection();
}

[shader("compute")]
void computeMain(uint tid : SV_DispatchThreadID)
{
    // Note: The original issue was due to "folding" of an instruction
    // into use sites as part of emitting high-level-language code.
    //
    // In this case, the initial value of `rayQuery` has only a single
    // use (in the call to `helper()`) and as a result was subject to
    // "folding" during emit, because it had no side-effects.

    RayQuery<0> rayQuery;
    let result = helper(rayQuery);

    gOutput[tid.x] = int(result.x);
}
