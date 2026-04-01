//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-compute -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -compute -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

int g(int arr[])
{
    int tmp[] = arr;
    return tmp.getCount();
}

int test(int arr[])
{
    return arr[0] + g(arr);
}

int test2<T>(inout T arr[])
{
    unmodified(arr);
    return arr.getCount();
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int arr[3] = {1, 2, 3};

    // CHECK: 4
    outputBuffer[0] = test(arr);

    // CHECK: 3
    outputBuffer[1] = test2(arr);

    int arr2[2] = { 1, 2 };

    // CHECK: 3
    outputBuffer[2] = test(arr2);

    // CHECK: 2
    outputBuffer[3] = test2(arr2);

}
