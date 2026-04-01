//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain 
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-cpu -compute -entry computeMain
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

static int myTwo = 2;
static int myThree = 1+2;

struct DefaultStruct_base
{
    int data0 = 1;
    int data1;

    __init()
    {
        data0 = 2;
        data1 = 3;
    }
};

struct DefaultStruct1 : DefaultStruct_base
{
    int data2 = 1;
    __init()
    {
        if (data0 == 2)
        {
            data2 = 4;
        }
    }
};

struct DefaultStruct2 : DefaultStruct_base
{
    __init()
    {
    }
};

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    DefaultStruct1 s1;
    DefaultStruct2 s2;

    // BUF: 1
    outputBuffer[0] = true
        && s1.data0 == 2
        && s1.data1 == 3
        && s1.data2 == 4
        && s2.data0 == 2
        && s2.data1 == 3
        ;
}
