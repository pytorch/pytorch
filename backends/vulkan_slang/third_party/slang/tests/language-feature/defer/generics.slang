//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IDroppable
{
    void drop(inout int i);
};

interface IDiscombobulatable
{
    void discombobulate(inout int i, int param);
}

void genericFunc<T>(int param)
    where T : IDroppable, IDiscombobulatable
{
    T t;
    int i = 0;
    defer t.drop(i);

    int p = param;
    defer t.discombobulate(i, p);
    t.discombobulate(i, p);
    p += 1;
}

struct TestType : IDroppable, IDiscombobulatable
{
    void drop(inout int i)
    {
        outputBuffer[i++] = 0xFF;
    }

    void discombobulate(inout int i, int param)
    {
        outputBuffer[i++] = param;
    }
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    genericFunc<TestType>(2);
}
