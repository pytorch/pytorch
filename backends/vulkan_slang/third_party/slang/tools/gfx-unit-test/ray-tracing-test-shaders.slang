// ray-tracing-test-shaders.slang

struct [raypayload] RayPayload
{
    float4 color : read(caller) : write(caller, closesthit, miss);
};

uniform RWTexture2D resultTexture;
uniform RaytracingAccelerationStructure sceneBVH;

[shader("raygeneration")]
void rayGenShaderA()
{
    int2 threadIdx = DispatchRaysIndex().xy;

    float3 rayDir = float3(0, 0, 1);
    float3 rayOrigin = 0;
    rayOrigin.x = (threadIdx.x * 2) - 1;
    rayOrigin.y = (threadIdx.y * 2) - 1;

    // Trace the ray.
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    RayPayload payload = { float4(0, 0, 0, 0) };
    TraceRay(sceneBVH, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);

    resultTexture[threadIdx.xy] = payload.color;
}

[shader("raygeneration")]
void rayGenShaderB()
{
    int2 threadIdx = DispatchRaysIndex().xy;

    float3 rayDir = float3(0, 0, 1);
    float3 rayOrigin = 0;
    rayOrigin.x = (threadIdx.x * 2) - 1;
    rayOrigin.y = (threadIdx.y * 2) - 1;

    // Trace the ray.
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    RayPayload payload = { float4(0, 0, 0, 0) };
    TraceRay(sceneBVH, RAY_FLAG_NONE, ~0, 1, 0, 1, ray, payload);

    resultTexture[threadIdx.xy] = payload.color;
}

[shader("miss")]
void missShaderA(inout RayPayload payload)
{
    payload.color = float4(1, 1, 1, 1);
}

[shader("miss")]
void missShaderB(inout RayPayload payload)
{
    payload.color = float4(0, 0, 0, 1);
}

[shader("closesthit")]
void closestHitShaderA(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    uint primitiveIndex = PrimitiveIndex();
    float4 color = float4(0, 0, 0, 1);
    color[primitiveIndex] = 1;
    payload.color = color;
}

[shader("closesthit")]
void closestHitShaderB(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    uint primitiveIndex = PrimitiveIndex();
    float4 color = float4(1, 1, 1, 1);
    color[primitiveIndex] = 0;
    payload.color = color;
}
