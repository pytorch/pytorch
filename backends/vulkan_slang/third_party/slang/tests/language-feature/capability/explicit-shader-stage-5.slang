//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -allow-glsl -profile sm_6_5

// Our code compiles here since we specify profile sm_6_5 (supports raytracing)
//CHECK: warning 36113
//CHECK: main
struct RayPayload
{
    uint data;
};

uniform RaytracingAccelerationStructure sceneBVH;
RWStructuredBuffer<uint> outputBuffer;
[shader("_raygen")]
void main()
{
    RayDesc ray;
    ray.Origin = float3(0,0,0);
    ray.Direction = float3(0,0,0);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    RayPayload payload = {};
    TraceRay(sceneBVH, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);

    outputBuffer[0] = payload.data;
}
