//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -O0

[[vk::push_constant]] float4 *positions;
[shader("vertex")]
float4 main()
{
    return positions[0];
}
// NOT %cbuffer__t = OpTypeStruct %_ptr_Function_v4float
// CHECK: %cbuffer__t = OpTypeStruct %_ptr_PhysicalStorageBuffer_v4float
