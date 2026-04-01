//TEST:SIMPLE(filecheck=CHECK_PASS): -target spirv -entry main -allow-glsl -profile sm_5_0
//TEST:SIMPLE(filecheck=CHECK_WARN): -target spirv -entry main -allow-glsl -profile sm_5_0 -DWARN

// Our code will stillwarn about spirv version being below 'spirv 1.3'. This is Because we are still using `Profile` internally.
// 'Profile' is an issue which needs to be refactored out since it only stores the info in a single integer through 'profileRaw'
//CHECK_PASS: warning 50011
//
// Our code does not warn about capabilities here since we specify profile 'sm_5_0' and a 'ser_motion_raygen' shader.
// 'ser_motion_raygen' implicitly uses "raygen + raytracing_motion + ser" capabilities.
//CHECK_PASS: main

// On the contrary, our code will warn without 'ser_motion_raygen'.
//CHECK_WARN: warning 41012 
//CHECK_WARN: warning 50011


struct RayPayload
{
    uint data;
};

uniform RaytracingAccelerationStructure sceneBVH;
RWStructuredBuffer<uint> outputBuffer;
#ifndef WARN
[shader("ser_motion_raygen")]
#else
[shader("raygen")]
#endif
void main()
{
    RayDesc ray;
    ray.Origin = float3(0,0,0);
    ray.Direction = float3(0,0,0);
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    RayPayload payload = {};
    TraceMotionRay(sceneBVH, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, 0, payload);

    outputBuffer[0] = payload.data;
}
