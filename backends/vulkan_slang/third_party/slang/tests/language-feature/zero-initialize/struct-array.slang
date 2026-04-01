//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -entry computeMain -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(smoke,compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain -allow-glsl -xslang -zero-initialize

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct MyStruct_base
{
    int a;
};
struct MyStruct : MyStruct_base
{
    int b;
    int c;
};
[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    
    Array<MyStruct, 3> myStructs;

// BUF: 1
    outputBuffer[0] = true
        && myStructs[0].a == 0
        && myStructs[0].b == 0
        && myStructs[0].c == 0

        && myStructs[1].a == 0
        && myStructs[1].b == 0
        && myStructs[1].c == 0

        && myStructs[2].a == 0
        && myStructs[2].b == 0
        && myStructs[2].c == 0
        ;
}
