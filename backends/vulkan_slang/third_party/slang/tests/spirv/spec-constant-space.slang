// Test that use of specialization constants does not cause space 0 to be reserved.

//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK-NOT: DescriptorSet 1

struct MyData { float4 val; RWStructuredBuffer<float> outputBuffer; }


[vk::specialization_constant]
const int kSpecializationConstant = 0;

[NumThreads(1,1,1)]
void main(ParameterBlock<MyData> g_data)
{
    g_data.outputBuffer[0] = g_data.val.x;
}