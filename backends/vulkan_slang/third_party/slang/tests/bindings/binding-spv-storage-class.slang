// binding-spv-storage-class.slang

//TEST:SIMPLE(filecheck=GL-SPIRV): -stage anyhit -entry main -target spirv-assembly -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=SPIRV): -stage anyhit -entry main -target spirv

// This test checks that the only the resource with Uniform, Storage or UniformConstant storage class can be decorated by binding or descriptor set.
struct MyStruct
{
    float3 org;
    float3 dir;
};

// ShaderRecordKHR storage class
layout(shaderRecordNV) ConstantBuffer<MyStruct> myStruct : register(b0, space1);

// Uniform buffer
ConstantBuffer<MyStruct> myStruct1 : register(b1, space1);

// Storage buffer
RWStructuredBuffer<MyStruct> myStruct2 : register(b2, space1);

// UniformConstant
Texture2D<float>    texture: register(b3, space1);
SamplerState        sampler: register(b4, space1);

[shader("anyhit")]
void main(out float3 pos)
{
    pos = myStruct.org + myStruct.dir +
          myStruct1.org + myStruct1.dir +
          myStruct2[0].org + myStruct2[0].dir;

    pos.x = texture.SampleLevel(
            sampler,
            pos.xy, 0);
}

// SPIRV-DAG: OpDecorate %myStruct1{{.*}} Binding 1
// SPIRV-DAG: OpDecorate %myStruct1{{.*}} DescriptorSet 1
// SPIRV-DAG: OpDecorate %myStruct2{{.*}} Binding 2
// SPIRV-DAG: OpDecorate %myStruct2{{.*}} DescriptorSet 1
// SPIRV-DAG: OpDecorate %texture{{.*}} Binding 3
// SPIRV-DAG: OpDecorate %texture{{.*}} DescriptorSet 1
// SPIRV-DAG: OpDecorate %sampler{{.*}} Binding 4
// SPIRV-DAG: OpDecorate %sampler{{.*}} DescriptorSet 1
//
//
// GL-SPIRV-DAG: OpDecorate %myStruct1{{.*}} DescriptorSet 1
// GL-SPIRV-DAG: OpDecorate %myStruct1{{.*}} Binding 1
// GL-SPIRV-DAG: OpDecorate %myStruct2{{.*}} DescriptorSet 1
// GL-SPIRV-DAG: OpDecorate %myStruct2{{.*}} Binding 2
// GL-SPIRV-DAG: OpDecorate %texture{{.*}} DescriptorSet 1
// GL-SPIRV-DAG: OpDecorate %texture{{.*}} Binding 3
// GL-SPIRV-DAG: OpDecorate %sampler{{.*}} DescriptorSet 1
// GL-SPIRV-DAG: OpDecorate %sampler{{.*}} Binding 4
