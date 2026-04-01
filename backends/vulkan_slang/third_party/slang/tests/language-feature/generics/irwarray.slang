//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

void writeToArray<U, T : IRWArray<U>>(inout T array, int index, U value) { array[index] = value; }
void writeToBuffer<U, T : IRWArray<U>>(T array, int index, U value) { array[index] = value; }
U readFromArray<U, T:IArray<U>>(T array, int index) { return array[index]; }

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    float arr[3] = { 1.0, 2.0, 3.0 };
    float4 v = float4(1.0, 2.0, 3.0, 4.0);
    float2x2 m = float2x2(1.0, 2.0, 3.0, 4.0);

    // CHECK: 1.0
    writeToBuffer(outputBuffer, 0, 1.0f);

    // CHECK: 4.0
    writeToArray(arr, 0, 4.0f);
    outputBuffer[1] = readFromArray(arr, 0);

    // CHECK: 3.0
    writeToArray(v, 3, 3.0f);
    outputBuffer[2] = readFromArray(v, 3);

    // CHECK: 30.0
    writeToArray(m, 1, float2(10.0f, 20.0f));
    outputBuffer[3] = readFromArray(m, 1).x + readFromArray(m, 1).y;

    writeToBuffer(outputBuffer, 0, readFromArray(outputBuffer, 0));
}
