
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -compute -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cuda -compute -output-using-type 
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -compute -output-using-type

int myIntrinsicFunc()
{
    __target_switch
    {
    case cpp:
    case hlsl:
    case cuda:
        __requirePrelude(R"(#define MY_CUDA_INTRINSIC 100)");
        __intrinsic_asm "(MY_CUDA_INTRINSIC)";
    }
}

int myIntrinsicFunc1()
{
    __target_switch
    {
    case cpp:
    case hlsl:
    case cuda:
        __requirePrelude(R"(#define MY_CUDA_INTRINSIC 100)");
        __intrinsic_asm "(MY_CUDA_INTRINSIC + 1)";
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK: 201
    outputBuffer[0] = myIntrinsicFunc() + myIntrinsicFunc1();
}
