//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=DX12):-slang -compute -dx12 -output-using-type -use-dxil -profile cs_6_5 -render-feature ray-query
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=VK):-vk -emit-spirv-directly -compute -shaderobj -output-using-type -render-feature ray-query

//TEST_INPUT: set accelStruct = AccelerationStructure
uniform RaytracingAccelerationStructure accelStruct;

float GetRayT<let RAY_QUERY_FLAGS: uint>(uint rayInlineFlags)
{
    RayDesc ray;
    ray.Origin = float3(0.1f, 0.1f, 0.0f);
    ray.Direction = float3(0.0f, 0.0f, 1.0f);
    ray.TMin = 0.01f;
    ray.TMax = 1e4f;

    RayQuery<RAY_QUERY_FLAGS> rq;
    rq.TraceRayInline(accelStruct, rayInlineFlags, 0xff, ray);
    bool proceed = rq.Proceed();

    if( proceed )
    {
        switch( rq.CandidateType() )
        {
            case CANDIDATE_NON_OPAQUE_TRIANGLE:
                rq.CommitNonOpaqueTriangleHit();
                rq.Abort();
                return rq.CommittedRayT();

            case CANDIDATE_PROCEDURAL_PRIMITIVE:
                rq.CommitProceduralPrimitiveHit(0.5f);
                rq.Abort();
                return rq.CommittedRayT();

            default:
                rq.Abort();
                return 0.0f;
        }
    }
    else 
    {
        if( rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT )
        {  
            rq.Abort();
            return rq.CommittedRayT();
        }
    }
    
    return 0.0f;
}

// Test the intrinsics that are available when RayQuery.Proceed() returns TRUE, as in the case of non-opaque triangles for example.
float CheckRayIntrinsics()
{
    RayDesc ray;
    ray.Origin = float3(0.1f, 0.1f, 0.0f);
    ray.Direction = float3(0.0f, 0.0f, 1.0f);
    ray.TMin = 0.0f;
    ray.TMax = 100.0f;

    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> rq;
    rq.TraceRayInline(accelStruct, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray);
    bool proceed = rq.Proceed();

    if( proceed )
    {
        if( rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE )
        {
            float score = 0.0f;

            rq.CommitNonOpaqueTriangleHit();
            rq.Abort();

            if( rq.RayFlags() == RAY_FLAG_FORCE_NON_OPAQUE )
                score += 1.0f;

            float3 wro = rq.WorldRayOrigin();
            if( wro.x == 0.1f && wro.y == 0.1f && wro.z == 0.0f )
                score += 1.0f;

            float3 wrd = rq.WorldRayDirection();
            if( wrd.x == 0.0f && wrd.y == 0.0f && wrd.z == 1.0f )
                score += 1.0f;

            if( rq.RayTMin() == 0.0f )
                score += 1.0f;

            if( rq.CandidateTriangleRayT() == 0.5f )
                score += 1.0f;

            if( rq.CommittedRayT() == 0.5f )
                score += 1.0f;

            if( rq.CandidateInstanceIndex() == 0 )
                score += 1.0f;

            if( rq.CandidateInstanceID() == 0 )
                score += 1.0f;

            if( rq.CandidateInstanceContributionToHitGroupIndex() == 0 )
                score += 1.0f;

            if( rq.CandidateGeometryIndex() == 0 )
                score += 1.0f;

            if( rq.CandidatePrimitiveIndex() == 0 )
                score += 1.0f;

            float3 oro = rq.CandidateObjectRayOrigin();
            if( oro.x == 0.1f && oro.y == 0.1f && oro.z == 0.0f )
                score += 1.0f;

            float3 ord = rq.CandidateObjectRayDirection();
            if( ord.x == 0.0f && ord.y == 0.0f && ord.z == 1.0f )
                score += 1.0f;

            float3x4 otwMat34 = rq.CandidateObjectToWorld3x4();
            score += otwMat34._11; score += otwMat34._12; score += otwMat34._13; score += otwMat34._14;
            score += otwMat34._21; score += otwMat34._22; score += otwMat34._23; score += otwMat34._24;
            score += otwMat34._31; score += otwMat34._32; score += otwMat34._33; score += otwMat34._34;

            float4x3 otwMat43 = rq.CandidateObjectToWorld4x3();
            score += otwMat43._11; score += otwMat43._12; score += otwMat43._13;
            score += otwMat43._21; score += otwMat43._22; score += otwMat43._23;
            score += otwMat43._31; score += otwMat43._32; score += otwMat43._33;
            score += otwMat43._41; score += otwMat43._42; score += otwMat43._43;

            float3x4 wtoMat34 = rq.CandidateWorldToObject3x4();
            score += wtoMat34._11; score += wtoMat34._12; score += wtoMat34._13; score += wtoMat34._14;
            score += wtoMat34._21; score += wtoMat34._22; score += wtoMat34._23; score += wtoMat34._24;
            score += wtoMat34._31; score += wtoMat34._32; score += wtoMat34._33; score += wtoMat34._34;

            float4x3 wtoMat43 = rq.CandidateWorldToObject4x3();
            score += wtoMat43._11; score += wtoMat43._12; score += wtoMat43._13;
            score += wtoMat43._21; score += wtoMat43._22; score += wtoMat43._23;
            score += wtoMat43._31; score += wtoMat43._32; score += wtoMat43._33;
            score += wtoMat43._41; score += wtoMat43._42; score += wtoMat43._43;

            if( rq.CommittedInstanceIndex() == 0 )
                score += 1.0f;

            if( rq.CommittedInstanceID() == 0 )
                score += 1.0f;

            if( rq.CommittedInstanceContributionToHitGroupIndex() == 0 )
                score += 1.0f;

            if( rq.CommittedGeometryIndex() == 0 )
                score += 1.0f;

            if( rq.CommittedPrimitiveIndex() == 0 )
                score += 1.0f;

            oro = rq.CommittedObjectRayOrigin();
            if( oro.x == 0.1f && oro.y == 0.1f && oro.z == 0.0f )
                score += 1.0f;

            ord = rq.CommittedObjectRayDirection();
            if( ord.x == 0.0f && ord.y == 0.0f && ord.z == 1.0f )
                score += 1.0f;

            otwMat34 = rq.CommittedObjectToWorld3x4();
            score += otwMat34._11; score += otwMat34._12; score += otwMat34._13; score += otwMat34._14;
            score += otwMat34._21; score += otwMat34._22; score += otwMat34._23; score += otwMat34._24;
            score += otwMat34._31; score += otwMat34._32; score += otwMat34._33; score += otwMat34._34;

            otwMat43 = rq.CommittedObjectToWorld4x3();
            score += otwMat43._11; score += otwMat43._12; score += otwMat43._13;
            score += otwMat43._21; score += otwMat43._22; score += otwMat43._23;
            score += otwMat43._31; score += otwMat43._32; score += otwMat43._33;
            score += otwMat43._41; score += otwMat43._42; score += otwMat43._43;

            wtoMat34 = rq.CommittedWorldToObject3x4();
            score += wtoMat34._11; score += wtoMat34._12; score += wtoMat34._13; score += wtoMat34._14;
            score += wtoMat34._21; score += wtoMat34._22; score += wtoMat34._23; score += wtoMat34._24;
            score += wtoMat34._31; score += wtoMat34._32; score += wtoMat34._33; score += wtoMat34._34;

            wtoMat43 = rq.CommittedWorldToObject4x3();
            score += wtoMat43._11; score += wtoMat43._12; score += wtoMat43._13;
            score += wtoMat43._21; score += wtoMat43._22; score += wtoMat43._23;
            score += wtoMat43._31; score += wtoMat43._32; score += wtoMat43._33;
            score += wtoMat43._41; score += wtoMat43._42; score += wtoMat43._43;

            float2 bary = rq.CandidateTriangleBarycentrics();
            if( bary.x != 0.0f && bary.y != 0.0f )
                score += 1.0f;

            if( rq.CandidateTriangleFrontFace() == true )
                score += 1.0f;

            bary = rq.CommittedTriangleBarycentrics();
            if( bary.x != 0.0f && bary.y != 0.0f )
                score += 1.0f;

            if( rq.CommittedTriangleFrontFace() == true )
                score += 1.0f;

            return score;
        }
    }

    return 0.0f;
}


//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int idx = dispatchThreadID.x;
    float val = 0.0f;

    const uint rayFlags[] = {
        RAY_FLAG_NONE,
        RAY_FLAG_FORCE_OPAQUE,
        RAY_FLAG_FORCE_NON_OPAQUE,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
        RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
        RAY_FLAG_CULL_OPAQUE,
        RAY_FLAG_CULL_NON_OPAQUE,
        RAY_FLAG_SKIP_TRIANGLES,
        RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES
    };
    const uint numRayFlags = sizeof(rayFlags) / sizeof(uint);

    // RAY_FLAG_FORCE_NON_OPAQUE, RAY_FLAG_CULL_OPAQUE, RAY_FLAG_SKIP_TRIANGLES returns 0 instead of 0.5
    for( uint i = 0; i < numRayFlags; ++i )
    {
        val += GetRayT<RAY_FLAG_NONE>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_FORCE_OPAQUE>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_FORCE_NON_OPAQUE>(rayFlags[i]); // RAY_FLAG_FORCE_NON_OPAQUE returns 0 for all RayInlineFlags
        val += GetRayT<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_CULL_BACK_FACING_TRIANGLES>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_CULL_FRONT_FACING_TRIANGLES>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_CULL_OPAQUE>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_CULL_NON_OPAQUE>(rayFlags[i]);
        val += GetRayT<RAY_FLAG_SKIP_TRIANGLES>(rayFlags[i]);
        if( rayFlags[i] != RAY_FLAG_SKIP_TRIANGLES )    // Skip this combo that's causing differences in the score between DX12 and VK
            val += GetRayT<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES>(rayFlags[i]);
    }

    val += CheckRayIntrinsics();

    outputBuffer[idx] = val;
}

// DX12: 88.5
// VK: 88.5