//TEST:SIMPLE(filecheck=SPIRV): -target spirv

//DISABLED_TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -output-using-type

// SPIRV: OpCapability VariablePointers

//TEST_INPUT:set buffer1 = ubuffer(data=[1.0], stride=4)
StructuredBuffer<float> buffer1;
//TEST_INPUT:set buffer2 = ubuffer(data=[2.0], stride=4)
StructuredBuffer<float> buffer2;
//TEST_INPUT:set output = out ubuffer(data=[0.0], stride=4)
RWStructuredBuffer<float> output;

[numthreads(2,1,1)]
void computeMain(int id : SV_DispatchThreadID)
{
    StructuredBuffer<float> buffer;
    if (id == 0)
        buffer = buffer1;
    else
        buffer = buffer2;
    output[id] = buffer[0];
    // CHECK: 1.0
    // CHECK: 2.0
}