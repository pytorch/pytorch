//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain 
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-cpu -compute -entry computeMain
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
import struct_field_initializer_import_target;
RWStructuredBuffer<int> outputBuffer;

void modifyOut(out DefaultStructNoInit2 noInit2)
{
    noInit2.data0 = 2;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    DefaultStructNoInit noInit = {};
    DefaultStructNoInit2 noInit2 = {};
    modifyOut(noInit2);
    // BUF: 1
    outputBuffer[0] = true
        && noInit.data0 == 2
        && noInit2.data0 == 2
        ;
}
