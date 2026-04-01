//TEST:SIMPLE(filecheck=MTL): -target metal
//TEST:SIMPLE(filecheck=MTLLIB): -target metallib
//TEST:SIMPLE(filecheck=CUDA): -target cuda -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=PTX): -target ptx -entry computeMain -stage compute 
//TEST:SIMPLE(filecheck=SPV): -target spirv -entry computeMain -stage compute -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=GLSL): -target glsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=HLSL): -target hlsl -entry computeMain -profile cs_6_6
//TEST:SIMPLE(filecheck=DXIL): -target dxil -entry computeMain -profile cs_6_6
//TEST:SIMPLE(filecheck=SPV): -target spirv

// SPV-DAG: %[[REG:[A-Za-z_0-9]+]] = OpAccessChain{{.*}}NonUniform
// SPV-DAG: %[[TEX:[A-Za-z_0-9]+]] = OpLoad {{.*}} %[[REG]]
// SPV-DAG: OpImageSampleExplicitLod {{.*}} %[[TEX]]

// HLSL: ResourceDescriptorHeap[NonUniformResourceIndex
// HLSL: SamplerDescriptorHeap[NonUniformResourceIndex(

// DXIL: define void @computeMain()

// GLSL: _slang_resource_heap{{[0-9_]+}}[nonuniformEXT

// MTL: void computeMain
// MTLLIB: @computeMain

// CUDA: tex2DLod
// PTX: computeMain

uniform StructuredBuffer<Sampler2D.Handle> t;
uniform RWStructuredBuffer<float4>.Handle buffer;

[numthreads(4,1,1)]
void computeMain(int3 id : SV_DispatchThreadID)
{
    (*buffer)[0] = nonuniform(t[id.x]).SampleLevel(float2(0.0), 0.0);
}