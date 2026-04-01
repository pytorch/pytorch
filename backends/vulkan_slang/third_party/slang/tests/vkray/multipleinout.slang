//TEST:SIMPLE(filecheck=CHECK): -stage closesthit -entry main -target spirv -emit-spirv-directly

// This test checks whether the spirv generated when there are multiple inout or out variables, they 
// all get consolidated into one IncomingRayPayloadKHR.

struct ReflectionRay
{
    float4 color;
};

StructuredBuffer<float4> colors;

[shader("closesthit")]
void main(
    BuiltInTriangleIntersectionAttributes attributes,
    inout ReflectionRay ioPayload,
    out float3 dummy)
{
    uint materialID = (InstanceIndex() << 1)
        + InstanceID()
        + PrimitiveIndex()
        + HitKind();

    ioPayload.color = colors[materialID];
    dummy = HitTriangleVertexPosition(0);
}

// CHECK: OpEntryPoint ClosestHitKHR %main "main" %{{.*}} %{{.*}} %gl_PrimitiveID %{{.*}} %gl_InstanceID %colors %{{.*}}
// CHECK: %_struct_{{.*}} = OpTypeStruct %ReflectionRay %v3float
// CHECK: %_ptr_IncomingRayPayloadKHR__struct_{{.*}} = OpTypePointer IncomingRayPayloadKHR %_struct_{{.*}}
// CHECK: %main = OpFunction %void None %{{.*}}
// CHECK: %materialID = OpIAdd %uint %{{.*}} %{{.*}}
// CHECK: %{{.*}} = OpAccessChain %_ptr_StorageBuffer_v4float %colors %int_0 %materialID
// CHECK: %{{.*}} = OpLoad %v4float %{{.*}}
// CHECK: %{{.*}} = OpAccessChain %_ptr_Input_v3float %{{.*}} %uint_0
// CHECK: %{{.*}} = OpLoad %v3float %{{.*}}
