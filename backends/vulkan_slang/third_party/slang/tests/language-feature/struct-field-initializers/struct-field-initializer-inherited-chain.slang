//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain 
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-cpu -compute -entry computeMain
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

static int myTwo = 2;
static int myThree = 1+2;

struct DefaultStruct_bottom
{
    int data0 = 1;
};
struct DefaultStruct_middle1 : DefaultStruct_bottom
{
    int data1 = 1;
};
struct DefaultStruct_middle2 : DefaultStruct_middle1
{
    int data2 = 1;
};
struct DefaultStruct_top : DefaultStruct_middle2
{
    int data3 = 1;
};

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    DefaultStruct_bottom s1 = {};
    DefaultStruct_middle1 s2 = {};
    DefaultStruct_middle2 s3 = {};
    DefaultStruct_top s4 = {};

    // BUF: 1
    outputBuffer[0] = true
        && s1.data0 == 1

        && s2.data0 == 1
        && s2.data1 == 1

        && s3.data0 == 1
        && s3.data1 == 1
        && s3.data2 == 1

        && s4.data0 == 1
        && s4.data1 == 1
        && s4.data2 == 1
        && s4.data3 == 1
        ;
}
