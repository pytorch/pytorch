// Test that push constants should not occupy the default
// space.

//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK-NOT: OpDecorate {{.*}} DescriptorSet 1

struct Data {
    StructuredBuffer<uint4> data;
    RWStructuredBuffer<float> output;
};
ParameterBlock<Data> gData;

[numthreads(1,1,1)]
void taskMain(uniform uint a)
{
    gData.output[0] = gData.data[0].x + a;
}
