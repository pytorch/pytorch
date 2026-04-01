//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -cpu
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -vk -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -cuda -compute
//TODO: metal is currently failing even with emulation, investigate.
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -mtl -compute -profile metallib_2_4
// No support for uint16_t on fxc - we need SM6.2 and dxil to use uint16_t with d3d12
// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -dx12 -profile cs_6_2 -use-dxil -shaderobj -render-feature hardware-device
// wgpu only has 32-bit support, so we do not try and test it here
// TODO: test GLSL pathway once emit-spirv-via-glsl is fixed and shader output reading is fixed for GLSL

//CHK:1

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint r1 = countbits(uint16_t(0b1U) << 8);
    uint2 r2 = countbits(uint16_t2(uint16_t(0b0U) << 8, uint16_t(0b1U) << 8));
    uint3 r3 = countbits(uint16_t3(uint16_t(0b0U) << 8, uint16_t(0b1U) << 8, uint16_t(0b11U) << 8));
    uint4 r4 = countbits(uint16_t4(uint16_t(0b0U) << 8, uint16_t(0b1U) << 8, uint16_t(0b11U) << 8, uint16_t(0b111U) << 8));

    uint r5 = countbits(int16_t(0b1) << 8);
    uint2 r6 = countbits(int16_t2(int16_t(0b0) << 8, int16_t(0b1) << 8));
    uint3 r7 = countbits(int16_t3(int16_t(0b0) << 8, int16_t(0b1) << 8, int16_t(0b11) << 8));
    uint4 r8 = countbits(int16_t4(int16_t(0b0) << 8, int16_t(0b1) << 8, int16_t(0b11) << 8, int16_t(0b111) << 8));

    uint16_t smallShiftU16 = uint16_t(0b111) << 16;
    int16_t smallShiftI16 = int16_t(0b1111) << 16;

    uint bitCountBigShiftU16 = countbits(smallShiftU16);
    uint bitCountBigShiftI16 = countbits(smallShiftI16);

    outputBuffer[0] = true
        && (r1 == 1)
        && (r2.x == 0 && r2.y == 1)
        && (r3.x == 0 && r3.y == 1 && r3.z == 2)
        && (r4.x == 0 && r4.y == 1 && r4.z == 2 && r4.w == 3)
        && (r5 == 1)
        && (r6.x == 0 && r6.y == 1)
        && (r7.x == 0 && r7.y == 1 && r7.z == 2)
        && (r8.x == 0 && r8.y == 1 && r8.z == 2 && r8.w == 3)
        && (bitCountBigShiftU16 == 0 && bitCountBigShiftI16 == 0)
        ;
}
