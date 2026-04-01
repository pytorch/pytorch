//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage anyhit -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry main -stage anyhit
//   Motion rays not supported on HLSL impl currently, so disable with NOMOTION define:
//TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly -entry main -stage anyhit -profile sm_6_5 -DNOMOTION

// SPIRV-NOT: OpCapability RayQueryKHR
// SPIRV-DAG: RayTracingKHR
// SPIRV-DAG: SPV_KHR_ray_tracing

// SPIRV: OpEntryPoint

// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn LaunchId{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn LaunchSize{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn WorldRayOrigin{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn WorldRayDirection{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn RayTmin{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn RayTmax{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn IncomingRayFlags{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn InstanceId
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn InstanceCustomIndex{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn RayGeometryIndexKHR
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn PrimitiveId
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn ObjectRayOrigin{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn ObjectRayDirection{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn ObjectToWorld{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn WorldToObject{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn HitKind{{NV|KHR}}

// SPIRV-DAG: OpTypePointer HitAttribute{{NV|KHR}}
// SPIRV-DAG: OpTypePointer IncomingRayPayload{{NV|KHR}}
 
// SPIRV-DAG: OpTerminateRayKHR
// SPIRV-DAG: OpIgnoreIntersectionKHR

// DXIL: main

// DXIL: call i32 @dx.op.dispatchRaysIndex.i32
// DXIL: call i32 @dx.op.dispatchRaysDimensions.i32

// DXIL: call float @dx.op.worldRayOrigin.f32
// DXIL: call float @dx.op.worldRayDirection.f32
// DXIL: call float @dx.op.rayTMin.f32
// DXIL: call float @dx.op.rayTCurrent.f32
// DXIL: call i32 @dx.op.rayFlags.i32

// DXIL: call i32 @dx.op.instanceIndex.i32
// DXIL: call i32 @dx.op.instanceID.i32
// DXIL: call i32 @dx.op.geometryIndex.i32
// DXIL: call i32 @dx.op.primitiveIndex.i32

// DXIL: call float @dx.op.objectRayOrigin.f32
// DXIL: call float @dx.op.objectRayDirection.f32
// DXIL: call float @dx.op.objectToWorld.f32
// DXIL: call float @dx.op.worldToObject.f32
// DXIL: call i32 @dx.op.hitKind.i32

// DXIL: call void @dx.op.acceptHitAndEndSearch
// DXIL: call void @dx.op.ignoreHit


struct RayPayload
{
    float RayHitT;
};

float CheckRayDispatchValues()
{
    float val = 0.0f;

    uint3 ri = DispatchRaysIndex();
    uint3 rd = DispatchRaysDimensions();

    val += ri.x; val += ri.y; val += ri.z;
    val += rd.x; val += rd.y; val += rd.z;

    return val;
}

float CheckRaySystemValues()
{
    float val = 0.0f;

    float3 wro = WorldRayOrigin();
    val += wro.x; val += wro.y; val += wro.z;

    float3 wrd = WorldRayDirection();
    val += wrd.x; val += wrd.y; val += wrd.z;

    float rayTMin = RayTMin();
    val += rayTMin;

    float rayTCurrent = RayTCurrent();
    val += rayTCurrent;

    uint rayFlags = RayFlags();
    switch (rayFlags)
    {
        case RAY_FLAG_NONE:
        case RAY_FLAG_FORCE_OPAQUE:
        case RAY_FLAG_FORCE_NON_OPAQUE:
        case RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH:
        case RAY_FLAG_SKIP_CLOSEST_HIT_SHADER:
        case RAY_FLAG_CULL_BACK_FACING_TRIANGLES:
        case RAY_FLAG_CULL_FRONT_FACING_TRIANGLES:
        case RAY_FLAG_CULL_OPAQUE:
        case RAY_FLAG_CULL_NON_OPAQUE:
        case RAY_FLAG_SKIP_TRIANGLES:
        case RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES:
            val += 1.0f;
            break;

        default:
            break;
    }

    return val;
}

float CheckObjectSpaceSystemValues()
{
    float val = 0.0f;

    val += InstanceIndex();
    val += InstanceID();
    val += GeometryIndex();
    val += PrimitiveIndex();

    float3 oro = ObjectRayOrigin();
    val += oro.x; val += oro.y; val += oro.z;

    float3 ord = ObjectRayDirection();
    val += ord.x; val += ord.y; val += ord.z;

    float3x4 f3x4 = ObjectToWorld3x4();
    val += f3x4._11; val += f3x4._12; val += f3x4._13; val += f3x4._14;
    val += f3x4._21; val += f3x4._22; val += f3x4._23; val += f3x4._24;
    val += f3x4._31; val += f3x4._32; val += f3x4._33; val += f3x4._34;

    float4x3 f4x3 = ObjectToWorld4x3();
    val += f4x3._11; val += f4x3._12; val += f4x3._13;
    val += f4x3._21; val += f4x3._22; val += f4x3._23;
    val += f4x3._31; val += f4x3._32; val += f4x3._33;
    val += f4x3._41; val += f4x3._42; val += f4x3._43;

    f3x4 = WorldToObject3x4();
    val += f3x4._11; val += f3x4._12; val += f3x4._13; val += f3x4._14;
    val += f3x4._21; val += f3x4._22; val += f3x4._23; val += f3x4._24;
    val += f3x4._31; val += f3x4._32; val += f3x4._33; val += f3x4._34;

    f4x3 = WorldToObject4x3();
    val += f4x3._11; val += f4x3._12; val += f4x3._13;
    val += f4x3._21; val += f4x3._22; val += f4x3._23;
    val += f4x3._31; val += f4x3._32; val += f4x3._33;
    val += f4x3._41; val += f4x3._42; val += f4x3._43;

    return val;
}

float CheckHitSpecificSystemValues()
{
    float val = 0.0f;

    uint hitKind = HitKind();
    if (hitKind == HIT_KIND_TRIANGLE_FRONT_FACE || hitKind == HIT_KIND_TRIANGLE_BACK_FACE)
    {
        val += 1.0f;
    }

    return val;
}

float CheckSysValueIntrinsics()
{
    float val = 0.0f;

    val += CheckRayDispatchValues();
    val += CheckRaySystemValues();
    val += CheckObjectSpaceSystemValues();
    val += CheckHitSpecificSystemValues();

    return val;
}

[shader("anyhit")]
void main( inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes )
{
    float val = 0.0f;

    val += CheckSysValueIntrinsics();

    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    val += barycentrics.x; val += barycentrics.y; val += barycentrics.z;

    float3 hitLocation = ObjectRayOrigin() + ObjectRayDirection() * RayTCurrent();

    if (hitLocation.x > 0.0f)
    {
        payload.RayHitT = val;
        AcceptHitAndEndSearch();
    }
    else if (hitLocation.y < 0.0f)
    {
        payload.RayHitT = val;
        IgnoreHit();
    }

    payload.RayHitT = val;
}
