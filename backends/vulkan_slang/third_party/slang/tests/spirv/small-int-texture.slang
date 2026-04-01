// Test that we can translate textures whose element type is a small integer type to valid spirv.

//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -output-using-type -emit-spirv-directly

//TEST_INPUT: RWTexture2D(format=R8Uint, size=4, content = one, mipMaps = 1):name g_Image
RWTexture2D<uint8_t> g_Image;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int id: SV_DispatchThreadID) {
    g_Image[id] = uint8_t(23);
    AllMemoryBarrier();
    // CHECK: 23
    outputBuffer[0] = g_Image[id];
}
