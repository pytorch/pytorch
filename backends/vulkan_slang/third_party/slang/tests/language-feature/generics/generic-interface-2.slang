//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

interface IFoo<let n: uint>
{
    static int sum(int arr[n]);
}

struct MyType<let n:uint> : IFoo<n>
{
    static int sum(int arr[n])
    {
        int rs = 0;
        for (int i = 0; i < n; i++)
            rs += arr[i];
        return rs;
    }
}

int test<let n:uint>(IFoo<n> foo)
{
    int arr[n];
    for (int i =0; i < n; i++)
        arr[i] = i;
    return foo.sum(arr);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    MyType<3> t;
    test(t);
    // CHECK: 3
    outputBuffer[0] = test(t);

    int arr[3] = {1,2,3};
    // CHECK: 3
    outputBuffer[1] = MyType<3>.sum(arr);
}
