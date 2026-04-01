//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry computeMain
// CHECK: {{.* }}= 0;
// CHECK-NOT: {{.* }}= 0;

//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly -allow-glsl
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -entry computeMain -allow-glsl
//DISABLE_TEST(smoke,compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain -allow-glsl -profile sm_6_2 -xslang -DDX12

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Base : IDefaultInitializable
{
    int data1;
};

struct idefault1_base : IDefaultInitializable
{
};
interface idefault2_base : IDefaultInitializable
{
};

struct idefault1 : idefault1_base = Base;

struct idefault2 : idefault2_base = Base;

idefault1 getDefault1()
{
    idefault1 default1;
    return default1;
}
idefault2 getDefault2()
{
    idefault2 default2;
    return default2;
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    // BUF: 1
    idefault1 default1 = getDefault1();
    idefault2 default2 = getDefault2();

    outputBuffer[0] = true
        && default1.inner.data1 == 0
        
        && default2.inner.data1 == 0
        ;
}
