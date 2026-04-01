//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -dx12
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu

// CHECK: 1
// CHECK-NEXT: 0
// CHECK-NEXT: 2
// CHECK-NEXT: 3

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IThing
{
    void writeInfo(int i);
}

enum FirstEnum
{
    A,
    B
};

enum SecondEnum
{
    C,
    D=3
};

extension FirstEnum: IThing
{
    void writeInfo(int i)
    {
        outputBuffer[i] = 1;
        outputBuffer[i+1] = this;
    }
}

extension SecondEnum: IThing
{
    void writeInfo(int i)
    {
        outputBuffer[i] = 2;
        outputBuffer[i+1] = this;
    }
}

void indirectionFunc<T: IThing>(T val, int i)
{
    val.writeInfo(i);
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    indirectionFunc(FirstEnum.A, 0);
    indirectionFunc(SecondEnum.D, 2);
}
