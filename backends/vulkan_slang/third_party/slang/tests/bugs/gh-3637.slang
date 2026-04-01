//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cuda -compute -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-cpu -compute -output-using-type

// CHECK: 9223372036854775808
// CHECK: 1
// CHECK: 18446744073709551615

// This tests exhibits a bug in constant folding in which the right shift
// operator unconditionally performs sign-extension.

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=8):out,name=outputBuffer
RWStructuredBuffer<uint64_t> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint64_t topBitSet = 0x8000000000000000ull;
    const uint64_t bottomBitSet = topBitSet >> 63;
    const uint64_t allBitsSet = int64_t(topBitSet) >> 63; // expect sign-extending which will set all bits.
    const uint64_t noBitsSet = topBitSet >> 64; // This exhibits undefined behaviour, as the compiler shifts right by at least the number of bits in the first operand
    outputBuffer[0] = topBitSet;
    outputBuffer[1] = bottomBitSet;
    outputBuffer[2] = allBitsSet;
    outputBuffer[3] = noBitsSet;
}
