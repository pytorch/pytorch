// rayquery-closesthit.slang
//TEST:SIMPLE(filecheck=CHECK): -profile glsl_460+GL_EXT_ray_tracing -stage closesthit -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK): -stage closesthit -entry main -target spirv-assembly -emit-spirv-directly

struct IntersectionPayload
{
    float3 triangleVerticeCommitted[3];
    float3 triangleVerticeCandidate[3];
};

RaytracingAccelerationStructure accelerationStructure;

void main(
	BuiltInTriangleIntersectionAttributes 	attributes,
	in out IntersectionPayload 				ioPayload)
{
	RayQuery<RAY_FLAG_NONE> rayQuery;

	uint instanceInclusionMask = 0x00;
	RayDesc rayDesc;
	rayQuery.TraceRayInline(accelerationStructure, RAY_FLAG_NONE, instanceInclusionMask, rayDesc);

	ioPayload.triangleVerticeCommitted = rayQuery.CandidateGetIntersectionTriangleVertexPositions();
	ioPayload.triangleVerticeCandidate = rayQuery.CommittedGetIntersectionTriangleVertexPositions();
}

// CHECK: OpCapability RayQueryKHR
// CHECK: OpCapability RayQueryPositionFetchKHR
// CHECK: OpExtension "SPV_KHR_ray_query"
// CHECK: OpExtension "SPV_KHR_ray_tracing_position_fetch"
// CHECK: OpEntryPoint ClosestHit{{NV|KHR}} %main "main"
// CHECK: OpRayQueryInitializeKHR %rayQuery{{.*}}
// CHECK: OpRayQueryGetIntersectionTriangleVertexPositionsKHR %_arr_v3float_{{u?}}int_3{{.*}} %rayQuery{{.*}} %{{u?}}int_0
// CHECK: OpRayQueryGetIntersectionTriangleVertexPositionsKHR %_arr_v3float_{{u?}}int_3{{.*}} %rayQuery{{.*}} %{{u?}}int_1
