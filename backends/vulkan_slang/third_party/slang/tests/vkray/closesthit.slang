// closesthit.slang
//TEST:SIMPLE(filecheck=CHECK): -profile glsl_460+GL_EXT_ray_tracing -stage closesthit -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK): -stage closesthit -entry main -target spirv-assembly -emit-spirv-directly

struct ReflectionRay
{
    float4 color;
    float3 dummyOut;
};

StructuredBuffer<float4> colors;

layout(shaderRecordNV)
cbuffer ShaderRecord
{
   uint shaderRecordID;
}

void main(
   BuiltInTriangleIntersectionAttributes attributes,
   in out ReflectionRay                  ioPayload)
{
   uint materialID = (InstanceIndex() << 1)
      + InstanceID()
      + PrimitiveIndex()
      + HitKind()
      + shaderRecordID;

   float4 color = colors[materialID];

   color *= RayTCurrent() - RayTMin();

   ioPayload.color = color;

   uint index = 0U;
   ioPayload.dummyOut = HitTriangleVertexPosition(index);

   index = 1U;
   ioPayload.dummyOut += HitTriangleVertexPosition(index);

   index = 2U;
   ioPayload.dummyOut += HitTriangleVertexPosition(index);
}

// CHECK: OpCapability RayTracing
// CHECK: OpCapability RayTracingPositionFetchKHR
// CHECK: OpEntryPoint ClosestHit{{NV|KHR}} %main "main"
// CHECK-DAG: OpDecorate %[[INSTANCE_ID:[A-Za-z0-9_]+]] BuiltIn InstanceId
// CHECK-DAG: OpDecorate %[[INSTANCE_INDEX:[A-Za-z0-9_]+]] BuiltIn InstanceCustomIndex{{NV|KHR}}
// CHECK-DAG: OpDecorate %{{.*}} BuiltIn PrimitiveId
// CHECK-DAG: OpDecorate %{{.*}} BuiltIn HitKind{{NV|KHR}}
// CHECK-DAG: OpDecorate %{{.*}} BuiltIn RayTmax{{NV|KHR}}
// CHECK-DAG: OpDecorate %{{.*}} BuiltIn RayTmin{{NV|KHR}}
// CHECK-DAG: OpDecorate %{{.*}} BuiltIn HitTriangleVertexPositionsKHR
// CHECK-DAG: %ShaderRecord{{.*}} = OpVariable %_ptr_ShaderRecordBuffer{{.*}} ShaderRecordBuffer{{NV|KHR}}
// CHECK-DAG: %{{.*}} = OpVariable %_ptr_IncomingRayPayload{{NV|KHR}}_ReflectionRay{{.*}} IncomingRayPayload{{NV|KHR}}
// CHECK-DAG: %{{.*}} = OpLoad %{{u?}}int %[[INSTANCE_ID]]
// CHECK-DAG: %{{.*}} = OpLoad %{{u?}}int %[[INSTANCE_INDEX]]
// CHECK-DAG: %{{.*}} = OpAccessChain %_ptr_ShaderRecordBuffer{{NV|KHR}}_uint %ShaderRecord{{.*}} %int_0
// CHECK-DAG: %{{.*}} = OpAccessChain %_ptr_IncomingRayPayload{{NV|KHR}}_v4float %{{.*}} %int_0
// CHECK-DAG: %{{.*}} = OpAccessChain %_ptr_Input_v3float %{{.*}} %{{u?}}int_0
// CHECK-DAG: %{{.*}} = OpAccessChain %_ptr_Input_v3float %{{.*}} %{{u?}}int_1
// CHECK-DAG: %{{.*}} = OpAccessChain %_ptr_Input_v3float %{{.*}} %{{u?}}int_2
