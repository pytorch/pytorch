//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -cpu
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -vk -compute -render-feature int64
// emit-spirv-via-glsl is currently ignored, but even working around this, output does not appear to be captured for GLSL
// No support for uint64_t in GLSL without an extension like GL_EXT_shader_explicit_arithmetic_types_int64
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -vk -compute -render-feature int64 -emit-spirv-via-glsl -profile GLSL_400 -Xslang... -capability GL_EXT_shader_explicit_arithmetic_types_int64.
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -cuda -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -mtl -compute
// No support for uint64_t on fxc - we need SM6.0 and dxil to use uint64_t with d3d12
// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -dx12 -profile cs_6_0 -use-dxil -shaderobj -render-feature hardware-device
// wgpu only has 32-bit support, so we do not try and test it here

//CHK:1

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint r1 = countbits(0b1ULL << 32);
    uint2 r2 = countbits(uint64_t2(0b0ULL << 32, 0b1ULL << 32));
    uint3 r3 = countbits(uint64_t3(0b0ULL << 32, 0b1ULL << 32, 0b11ULL << 32));
    uint4 r4 = countbits(uint64_t4(0b0ULL << 32, 0b1ULL << 32, 0b11ULL << 32, 0b111ULL << 32));

    uint r5 = countbits(0b1LL << 32);
    uint2 r6 = countbits(int64_t2(0b0LL << 32, 0b1LL << 32));
    uint3 r7 = countbits(int64_t3(0b0LL << 32, 0b1LL << 32, 0b11LL << 32));
    uint4 r8 = countbits(int64_t4(0b0LL << 32, 0b1LL << 32, 0b11LL << 32, 0b111LL << 32));

    uint bigShiftU32 = 0b111U << 32;
    int bigShiftI32 = 0b1111 << 32;

    uint bitCountBigShiftU32 = countbits(bigShiftU32);
    uint bitCountBigShiftI32 = countbits(bigShiftI32);

    outputBuffer[0] = true
        && (r1 == 1)
        && (r2.x == 0 && r2.y == 1)
        && (r3.x == 0 && r3.y == 1 && r3.z == 2)
        && (r4.x == 0 && r4.y == 1 && r4.z == 2 && r4.w == 3)
        && (r5 == 1)
        && (r6.x == 0 && r6.y == 1)
        && (r7.x == 0 && r7.y == 1 && r7.z == 2)
        && (r8.x == 0 && r8.y == 1 && r8.z == 2 && r8.w == 3)
        && (bitCountBigShiftU32 == 0 && bitCountBigShiftI32 == 0)
        ;
}
