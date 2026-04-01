//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -cpu
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -vk -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -cuda -compute
//TODO: metal is currently failing even with emulation, investigate.
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -mtl -compute -profile metallib_2_4
// Not testing the following:
// -dx12/hlsl, No support for uint8_t with hlsl
// -wgpu, only has 32-bit support
// -vk/glsl, No support for uint8_t with glsl

//CHK:1

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint r1 = countbits(uint8_t(0b1U) << 4);
    uint2 r2 = countbits(uint8_t2(uint8_t(0b0U) << 4, uint8_t(0b1U) << 4));
    uint3 r3 = countbits(uint8_t3(uint8_t(0b0U) << 4, uint8_t(0b1U) << 4, uint8_t(0b11U) << 4));
    uint4 r4 = countbits(uint8_t4(uint8_t(0b0U) << 4, uint8_t(0b1U) << 4, uint8_t(0b11U) << 4, uint8_t(0b111U) << 4));

    uint r5 = countbits(int8_t(0b1) << 4);
    uint2 r6 = countbits(int8_t2(int8_t(0b0) << 4, int8_t(0b1) << 4));
    uint3 r7 = countbits(int8_t3(int8_t(0b0) << 4, int8_t(0b1) << 4, int8_t(0b11) << 4));
    uint4 r8 = countbits(int8_t4(int8_t(0b0) << 4, int8_t(0b1) << 4, int8_t(0b11) << 4, int8_t(0b111) << 4));

    uint8_t smallShiftU8 = uint8_t(0b111) << 8;
    int8_t smallShiftI8 = int8_t(0b1111) << 8;

    uint bitCountBigShiftU8 = countbits(smallShiftU8);
    uint bitCountBigShiftI8 = countbits(smallShiftI8);

    outputBuffer[0] = true
        && (r1 == 1)
        && (r2.x == 0 && r2.y == 1)
        && (r3.x == 0 && r3.y == 1 && r3.z == 2)
        && (r4.x == 0 && r4.y == 1 && r4.z == 2 && r4.w == 3)
        && (r5 == 1)
        && (r6.x == 0 && r6.y == 1)
        && (r7.x == 0 && r7.y == 1 && r7.z == 2)
        && (r8.x == 0 && r8.y == 1 && r8.z == 2 && r8.w == 3)
        && (bitCountBigShiftU8 == 0 && bitCountBigShiftI8 == 0)
        ;
}
