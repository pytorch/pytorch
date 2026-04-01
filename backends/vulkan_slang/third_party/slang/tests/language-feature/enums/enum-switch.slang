//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

enum class E : uint32_t
{
    A,
    B,
    C
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[NumThreads(1,1,1)]
void computeMain(int i : SV_DispatchThreadID)
{
    E e = E(i);
    switch (e)
    {
    case E::A:
        outputBuffer[0] = 1;
        break;
    case E::B:
        outputBuffer[0] = 2;
        break;
    case E::C:
        outputBuffer[0] = 3;
        break;
    }
    // CHECK: 1
}