//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

T sum<T:IFloat>(IArray<T> array)
{
    T result = T(0);
    for (int i = 0; i < array.getCount(); i++)
    {
        result = result + array[i];
    }
    return result;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    float arr[3] = { 1.0, 2.0, 3.0 };
    float4 v = float4(1.0, 2.0, 3.0, 4.0);
    float2x2 m = float2x2(1.0, 2.0, 3.0, 4.0);
    outputBuffer[0] = sum(arr);
    outputBuffer[1] = sum(v);
    outputBuffer[2] = sum(sum(m));
    // CHECK: 6.0
    // CHECK: 10.0
    // CHECK: 10.0
}
