//TEST:SIMPLE(filecheck=SPV): -target spirv

// SPV: OpDecorate %resourceHeap DescriptorSet 10
// SPV: OpAccessChain {{.*}} %resourceHeap
// SPV: OpImageSample

uniform StructuredBuffer<DescriptorHandle<Sampler2D>> t;
uniform DescriptorHandle<RWStructuredBuffer<float4>> buffer;

[vk::binding(0, 10)]
__DynamicResource<__DynamicResourceKind.General> resourceHeap[]; 

// A customized function that overrides the default behavior of fetch texture resources.
export T getDescriptorFromHandle<T:IOpaqueDescriptor>(DescriptorHandle<T> handleValue)
{
    __target_switch
    {
    case spirv:
        if (T.kind != DescriptorKind.Buffer)
            return resourceHeap[((uint2)handleValue).x].asOpaqueDescriptor<T>();
        else
            return defaultGetDescriptorFromHandle(handleValue);
    default:
        return defaultGetDescriptorFromHandle(handleValue);
    }
}

[numthreads(1,1,1)]
void computeMain()
{
    buffer[0] = t[0]->SampleLevel(float2(0.0), 0.0);
}