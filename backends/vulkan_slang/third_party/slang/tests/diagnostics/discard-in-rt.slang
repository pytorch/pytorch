//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: 'closestHit' uses features that are not available in 'closesthit' stage.
// CHECK: see using of 'discard'

struct PrimaryRayPayload{}
[require(spvRayTracingKHR)]
[shader("closesthit")]
void closestHit(
    inout PrimaryRayPayload rayData : SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    discard;
}

// CHECK: 'closestHit1' uses features that are not available in 'closesthit' stage
// CHECK: see using of 'discard'
[shader("closesthit")]
void closestHit1(
    inout PrimaryRayPayload rayData : SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    discard;
}
