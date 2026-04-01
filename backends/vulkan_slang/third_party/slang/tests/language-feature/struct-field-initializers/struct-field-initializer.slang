//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain 
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-cpu -compute -entry computeMain
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

static int myTwo = 2;
static int myThree = 1+2;

struct DefaultStructNoInit
{
    int data0 = 0;
    int data1 = myTwo;
    int data2 = 2;
};
struct DefaultStructWithInit
{
    int data0 = 3;
    int data1 = myThree;
    int data2;
    __init()
    {
        data2 = 3;
    }
};
struct DefaultStructWithInit2
{
    int data0 = 4;
    int data1 = 1;
    int data2 = 1;
    __init()
    {
        data1 = 4;
        data2 = 4;
    }
};
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    DefaultStructNoInit noInit = DefaultStructNoInit();
    noInit.data0 = 2;
    DefaultStructWithInit withInit = DefaultStructWithInit();
    DefaultStructWithInit2 withInit2 = DefaultStructWithInit2();
    // BUF: 1
    outputBuffer[0] = true
        && noInit.data0 == 2
        && noInit.data1 == 2
        && noInit.data2 == 2

        && withInit.data0 == 3
        && withInit.data1 == 3
        && withInit.data2 == 3

        && withInit2.data0 == 4
        && withInit2.data1 == 4
        && withInit2.data2 == 4
        ;
}
