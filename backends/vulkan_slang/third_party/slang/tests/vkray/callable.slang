// callable.slang

//TEST:SIMPLE(filecheck=CHECK): -stage callable -entry main -target spirv-assembly -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=CHECK): -stage callable -entry main -target spirv

import callable_shared;

Texture2D gAlbedoMap;
SamplerState gSampler;

void main(in out MaterialPayload ioPayload)
{
	ioPayload.albedo = gAlbedoMap.SampleLevel(
		gSampler,
		ioPayload.uv,
		0);
}

// CHECK-DAG: OpCapability RayTracingKHR
// CHECK-DAG: OpExtension "SPV_KHR_ray_tracing"
// CHECK: OpEntryPoint
// CHECK-DAG: OpTypePointer IncomingCallableData{{NV|KHR}}
// CHECK-DAG: OpTypePointer IncomingCallableData{{NV|KHR}}
// CHECK-DAG: OpTypePointer IncomingCallableData{{NV|KHR}}
// CHECK-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}
// CHECK-DAG: %{{.*}} = OpAccessChain %{{.*}} %{{.*}} %{{.*}}
