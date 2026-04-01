//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile lib_6_7 -stage raygeneration -entry rayGenShaderA
//TEST:SIMPLE(filecheck=DXIL): -target dxil -profile lib_6_7 -stage raygeneration -entry rayGenShaderA

// CHECK: struct [raypayload]
// CHECK: float4 color_0 : read(caller, anyhit) : write(caller);
// DXIL: define void @
// DXIL: !dx.dxrPayloadAnnotations

struct [raypayload] RayPayload
{
    float4 color : read(caller, anyhit) : write(caller);
};

uniform RWTexture2D<float4> resultTexture;
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
