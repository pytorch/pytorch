//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry computeMain -zero-initialize
// CHECK-COUNT-3: {{.* }}= 0;

//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -xslang -zero-initialize
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
MyStruct getStruct()
{
    MyStruct myStruct;
    return myStruct;
}
[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    MyStruct myStruct = getStruct();

// BUF: 1
    outputBuffer[0] = true
        && myStruct.a == 0
        && myStruct.b == 0
        && myStruct.c == 0
        ;
}
