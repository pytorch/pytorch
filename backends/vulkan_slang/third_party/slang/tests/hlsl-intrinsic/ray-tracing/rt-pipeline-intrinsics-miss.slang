//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage miss -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry main -stage miss
//   Motion rays not supported on HLSL impl currently, so disable with NOMOTION define:
//TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly -entry main -stage miss -profile sm_6_5 -DNOMOTION

// SPIRV-NOT: OpCapability RayQueryKHR
// SPIRV-DAG: OpCapability RayTracingKHR
// SPIRV-DAG: OpCapability RayTracingMotionBlurNV
// SPIRV-DAG: SPV_KHR_ray_tracing
// SPIRV-DAG: SPV_NV_ray_tracing_motion_blur

// SPIRV: OpEntryPoint

// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn LaunchId{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn LaunchSize{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} Binding 0
// SPIRV-DAG: OpDecorate %{{.*}} DescriptorSet 0
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn WorldRayOrigin{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn WorldRayDirection{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn RayTmin{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn RayTmax{{NV|KHR}}
// SPIRV-DAG: OpDecorate %{{.*}} BuiltIn IncomingRayFlags{{NV|KHR}}

// SPIRV: OpTraceRayKHR
// SPIRV: OpTraceRayMotionNV
// SPIRV: OpExecuteCallableKHR

// DXIL: main

// DXIL: call i32 @dx.op.dispatchRaysIndex.i32
// DXIL: call i32 @dx.op.dispatchRaysDimensions.i32
// DXIL: call void @dx.op.traceRay.struct.RayPayload_0

// DXIL: call float @dx.op.worldRayOrigin.f32
// DXIL: call float @dx.op.worldRayDirection.f32
// DXIL: call float @dx.op.rayTMin.f32
// DXIL: call float @dx.op.rayTCurrent.f32
// DXIL: call i32 @dx.op.rayFlags.i32

// DXIL: call void @dx.op.callShader.struct.CallableParams_0

[[vk::binding(0)]]
uniform RaytracingAccelerationStructure accelStruct : register(t0);

struct RayPayload
{
    float RayHitT;
};

struct CallableParams
{
    float value;
};

#define T_MIN 0.01f
#define T_MAX 1e4f

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

float CheckTraceRay(RayPayload payload, RayDesc rayDesc)
{
    float val = 0.0f;
    float currentTime = 1;

    TraceRay(accelStruct, RAY_FLAG_NONE, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_FORCE_OPAQUE, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_FORCE_NON_OPAQUE, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_CULL_OPAQUE, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_CULL_NON_OPAQUE, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_SKIP_TRIANGLES, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;
    TraceRay(accelStruct, RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, ~0, 0, 1, 0, rayDesc, payload);
    val += payload.RayHitT;

#if !defined(NOMOTION)
    TraceMotionRay(accelStruct, RAY_FLAG_NONE, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_FORCE_OPAQUE, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_FORCE_NON_OPAQUE, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_CULL_OPAQUE, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_CULL_NON_OPAQUE, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_SKIP_TRIANGLES, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
    TraceMotionRay(accelStruct, RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES, ~0, 0, 1, 0, rayDesc, currentTime, payload);
    val += payload.RayHitT;
#endif

    return val;
}

float CheckSysValueIntrinsics()
{
    float val = 0.0f;

    val += CheckRayDispatchValues();
    val += CheckRaySystemValues();

    return val;
}

[shader("miss")]
void main( inout RayPayload payload )
{
    float2 dir = (DispatchRaysIndex().xy / DispatchRaysDimensions().xy) * 2.0f - 1.0f;
    float aspectRatio = DispatchRaysDimensions().x / DispatchRaysDimensions().y;

    RayDesc rayDesc;
    rayDesc.Origin = float3(0.0f, 0.0f, 0.0f);
    rayDesc.Direction = normalize(float3(dir.x * aspectRatio, -dir.y, 1)); ;
    rayDesc.TMin = T_MIN;
    rayDesc.TMax = T_MAX;

    //RayPayload payload;
    payload.RayHitT = T_MAX;

    float val = 0.0f;

    val += CheckTraceRay(payload, rayDesc);

    val += CheckSysValueIntrinsics();

    CallableParams params;
    CallShader( 0, params );
    val += params.value;

    payload.RayHitT = val;
}
