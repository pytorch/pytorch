//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int32_t> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    let result = CoopVec<int32_t, 4>(1,2,3,4,5);
    // CHECK: error 41400: static assertion failed, number of arguments to CoopVec constructor must match number of elements

    for(int i = 0; i < result.getCount(); ++i)
        outputBuffer[i] = result[i];;
}
