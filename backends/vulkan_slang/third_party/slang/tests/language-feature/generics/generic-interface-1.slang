//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

interface IEqlTestable<T>
{
    bool testEql(T v1);
}

bool test<T>(IEqlTestable<T> v0, T v1)
{
    return v0.testEql(v1);
}

struct MyType : IEqlTestable<MyType>
{
    int val;
    bool testEql(MyType v1)
    {
        return val == v1.val;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(2, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    MyType obj1, obj2;
    obj1.val = tid;
    obj2.val = 1;
    let result = test(obj1, obj2);
    outputBuffer[tid] = result ? 1 : 0;
    // CHECK: 0
    // CHECK: 1
}
