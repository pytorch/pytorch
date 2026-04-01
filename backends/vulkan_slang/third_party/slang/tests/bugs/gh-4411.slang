//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -dx12 -use-dxil -shaderobj

//TEST_INPUT:RWTexture1D(format=R32Uint, size=8, content = zero, mipMaps = 1):name=texture1D
RWTexture1D<uint> texture1D;

//TEST_INPUT:RWTexture2D(format=R32Uint, size=8, content = zero, mipMaps = 1):name=texture2D
RWTexture2D<uint> texture2D;

//TEST_INPUT:ubuffer(data=[0]):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

void computeMain() {
    texture1D[1].x = 15;
    texture2D[uint2(2, 3)].x = 16;

    outputBuffer[0] = uint(true
            && texture1D[1].x == 15
            && texture2D[uint2(2, 3)].x == 16
        );
    //BUF: 1
}
