//TEST:SIMPLE(filecheck=SPV): -target spirv -bindless-space-index 101

// SPV: OpDecorate %__slang_resource_heap{{.*}} Binding 0
// SPV: OpDecorate %__slang_resource_heap{{.*}} DescriptorSet 101
// SPV: OpImageSample

uniform StructuredBuffer<DescriptorHandle<Sampler2D>> t;
uniform DescriptorHandle<RWStructuredBuffer<float4>> buffer;

[numthreads(1,1,1)]
void computeMain()
{
    (*buffer)[0] = t[0].SampleLevel(float2(0.0), 0.0);
}