//TEST:SIMPLE(filecheck=CUDA): -target cuda -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=MTL): -target metallib -entry computeMain -profile cs_6_6
//TEST:SIMPLE(filecheck=HLSL): -target hlsl -entry computeMain -profile cs_6_6
//TEST:SIMPLE(filecheck=DXIL): -target dxil -entry computeMain -profile cs_6_6
//TEST:SIMPLE(filecheck=SPV): -target spirv

// SPV: OpImageSample
// HLSL: ResourceDescriptorHeap[{{.*}}]
// HLSL: SamplerDescriptorHeap[{{.*}}]
// HLSL: SampleLevel
// DXIL: computeMain
// CUDA: computeMain
// MTL: define void @computeMain

uniform StructuredBuffer<DescriptorHandle<Sampler2D>> t;
uniform DescriptorHandle<RWStructuredBuffer<float4>> buffer;

[numthreads(1,1,1)]
void computeMain()
{
    buffer[0] = t[0]->SampleLevel(float2(0.0), 0.0);
}