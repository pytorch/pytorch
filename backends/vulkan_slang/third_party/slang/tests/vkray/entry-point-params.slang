// entry-point-params.slang

// Test that entry-point `uniform` parameters on ray-tracing
// shaders properly map to the "shader record" in SPIR-V output.

//TEST:SIMPLE(filecheck=CHECK): -profile glsl_460+spirv_1_4 -stage raygeneration -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK): -stage raygeneration -entry main -target spirv-assembly -emit-spirv-directly

RWStructuredBuffer<float> buffer;

void main(
	uniform float value) 
{
	buffer[DispatchRaysIndex().x] = value;
}

// CHECK: OpCapability RayTracingKHR
// CHECK: OpExtension "SPV_KHR_ray_tracing"
// CHECK: OpEntryPoint RayGeneration{{NV|KHR}} %main
// CHECK-DAG: OpDecorate %[[LaunchID:[A-Za-z0-9_]+]] BuiltIn LaunchId{{NV|KHR}}
// CHECK-DAG: %[[LaunchID]] = OpVariable %_ptr_Input_v3uint Input
// CHECK-DAG: %{{.*}} = OpLoad %v3uint %[[LaunchID]]
// CHECK-DAG: %{{.*}} = OpAccessChain %_ptr_ShaderRecordBuffer{{NV|KHR}}_float %{{.*}} %int_0
