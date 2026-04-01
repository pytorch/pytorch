// rayquery-compute.slang
//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry main -target spirv-assembly -emit-spirv-directly

RaytracingAccelerationStructure accelerationStructure;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	RayQuery<RAY_FLAG_NONE> rayQuery;

	uint instanceInclusionMask = 0x00;
	RayDesc rayDesc;
	rayQuery.TraceRayInline(accelerationStructure, RAY_FLAG_NONE, instanceInclusionMask, rayDesc);
}

// CHECK: OpCapability RayQueryKHR
// CHECK-NOT: OpCapability RayTracingKHR
// CHECK: OpExtension "SPV_KHR_ray_query"
// CHECK-NOT: OpExtension "SPV_KHR_ray_tracing"
