//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry computeMain
// CHECK-COUNT-6: {{.* }}= 0U;

//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly -allow-glsl
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -entry computeMain -allow-glsl
//DISABLE_TEST(smoke,compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain -allow-glsl -profile sm_6_2 -xslang -DDX12

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct idefault1 : IDefaultInitializable
{
    uint data;
};

struct idefault2_base : IDefaultInitializable
{
    uint data1;
};
struct idefault2 : idefault2_base
{
    uint data2 = 1;
};

interface idefault3_base : IDefaultInitializable
{
};
struct idefault3 : idefault3_base
{
    uint data;
};

struct idefault4
{
    uint data;
};

extension idefault4 : IDefaultInitializable
{
}

struct idefault5_base : IDefaultInitializable
{
    uint data1;
};

struct idefault5 : idefault5_base
{
    uint data2;
};

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
idefault3 getDefault3()
{
    idefault3 default3;
    return default3;
}
idefault4 getDefault4()
{
    idefault4 default4;
    return default4;   
}

idefault5 getDefault5()
{
    idefault5 default5;
    return default5;
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    // BUF: 1
    idefault1 default1 = getDefault1();
    idefault2 default2 = getDefault2();
    idefault3 default3 = getDefault3();
    idefault4 default4 = getDefault4();
    idefault5 default5 = getDefault5();

    outputBuffer[0] = true
        && default1.data == 0

        && default2.data1 == 0
        && default2.data2 == 1

        && default3.data == 0

        && default4.data == 0

        && default5.data1 == 0
        && default5.data2 == 0
        ;
}
