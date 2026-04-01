//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK-DAG: %[[cbuffer__t:[A-Za-z0-9_]+]] = OpTypeStruct %_ptr_PhysicalStorageBuffer_uint 
// CHECK-DAG: %light_buffer = OpVariable %_ptr_PushConstant_[[cbuffer__t]] PushConstant

// CHECK: OpAccessChain %_ptr_PushConstant
// CHECK-NEXT: OpLoad
// CHECK-NEXT: OpBitcast %_ptr_PhysicalStorageBuffer

struct LightBuffer {
    uint8_t lights[];
}

[vk::push_constant]
LightBuffer* light_buffer;

[shader("vertex")]
float4 vertMain() : SV_Position {
    return float4(light_buffer.lights[0]);
}

[shader("fragment")]
float4 fragMain() : COLOR0 {
    return float4(1.0);
}