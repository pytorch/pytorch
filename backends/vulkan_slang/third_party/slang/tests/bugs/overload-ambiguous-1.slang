// https://github.com/shader-slang/slang/issues/4476

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cuda -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

namespace A1
{
    uint func()
    {
        return 1u;
    }

    namespace A2
    {
        uint func()
        {
            return 2u;
        }

        namespace A3
        {
            uint func()
            {
                return 3u;
            }

            uint test2()
            {
                return func();      // choose A3::func()
            }
        }

        namespace A4
        {
            uint test()
            {
                return func();      // choose A2::func()
            }
        }
    }
}

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain(uint3 threadID: SV_DispatchThreadID)
{
    using namespace A1;
    using namespace A1::A2;
    using namespace A1::A2::A3;
    using namespace A1::A2::A4;
    outputBuffer[0] = test();
    // BUF: 2

    outputBuffer[1] = func(); // choose the A1::func()
    // BUF-NEXT: 1

    outputBuffer[2] = test2();
    // BUF-NEXT: 3
}
