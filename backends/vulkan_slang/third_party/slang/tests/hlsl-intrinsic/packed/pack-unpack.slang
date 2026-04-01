//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -profile cs_6_6 -dx12 -use-dxil -shaderobj -render-feature hardware-device
//TEST(compute):COMPARE_COMPUTE_EX:-metal -compute -shaderobj 
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

// 16 bit variants are not supported by WGSL.
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj -xslang -DWGSL
// Debug info for inlining errors can be given out, so disable them for this test.
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -g0

//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -xslang -DUSE_SLANG_SYNTAX
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -profile cs_6_6 -dx12 -use-dxil -shaderobj -render-feature hardware-device -xslang -DUSE_SLANG_SYNTAX
//TEST(compute):COMPARE_COMPUTE_EX:-metal -compute -shaderobj -xslang -DUSE_SLANG_SYNTAX
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj -xslang -DUSE_SLANG_SYNTAX

// 16 bit variants are not supported by WGSL.
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj -xslang -DWGSL -xslang -DUSE_SLANG_SYNTAX
// Debug info for inlining errors can be given out, so disable them for this test.
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -g0 -xslang -DUSE_SLANG_SYNTAX


//TEST_INPUT:ubuffer(data=[0xD37A83FF], stride=4):name unpackTestBuffer
StructuredBuffer<uint32_t> unpackTestBuffer;

//TEST_INPUT:ubuffer(data=[0xB3F 0x6A 0x123 0xD4], stride=4): name packTestBuffer
StructuredBuffer<uint32_t4> packTestBuffer;

// These should clamp to (5, 255, 255, 254) or (0x5, 0xFF, 0xFF, 0xFE).
//TEST_INPUT:ubuffer(data=[5 256 12345 254], stride=4): name packClampUTestBuffer
StructuredBuffer<int32_t4> packClampUTestBuffer;

// These should clamp to (-1, 127, -128, 125) or (0xFF, 0x7F, 0x80, 0x81)
// Inputs are [-1 250 -32768 -127].
//TEST_INPUT:ubuffer(data=[0xFFFFFFFF 0xFA 0xFFFF8000 0xFFFFFF81], stride=4): name packClampSTestBuffer
StructuredBuffer<int32_t4> packClampSTestBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

uint32_t4 __unpack_u8u32(uint value)
{
#if defined(USE_SLANG_SYNTAX)
    return unpackUint4x8ToUint32(value);
#else
    return unpack_u8u32(uint8_t4_packed(value));
#endif
}

uint16_t4 __unpack_u8u16(uint value)
{
#if defined(USE_SLANG_SYNTAX)
    return unpackUint4x8ToUint16(value);
#else
    return unpack_u8u16(uint8_t4_packed(value));
#endif
}

int32_t4 __unpack_s8s32(uint value)
{
#if defined(USE_SLANG_SYNTAX)
    return unpackInt4x8ToInt32(value);
#else
    return unpack_s8s32(int8_t4_packed(value));
#endif
}

int16_t4 __unpack_s8s16(uint value)
{
#if defined(USE_SLANG_SYNTAX)
    return unpackInt4x8ToInt16(value);
#else
    return unpack_s8s16(int8_t4_packed(value));
#endif
}

uint __pack_u8(uint32_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packUint4x8(value);
#else
    return uint(pack_u8(value));
#endif
}

uint __pack_u8(uint16_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packUint4x8(value);
#else
    return uint(pack_u8(value));
#endif
}

uint __pack_s8(int32_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packInt4x8(value);
#else
    return uint(pack_s8(value));
#endif
}

uint __pack_s8(int16_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packInt4x8(value);
#else
    return uint(pack_s8(value));
#endif
}

uint __pack_clamp_u8(int32_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packUint4x8Clamp(value);
#else
    return uint(pack_clamp_u8(value));
#endif
}

uint __pack_clamp_u8(int16_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packUint4x8Clamp(value);
#else
    return uint(pack_clamp_u8(value));
#endif
}

uint __pack_clamp_s8(int32_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packInt4x8Clamp(value);
#else
    return uint(pack_clamp_s8(value));
#endif
}

uint __pack_clamp_s8(int16_t4 value)
{
#if defined(USE_SLANG_SYNTAX)
    return packInt4x8Clamp(value);
#else
    return uint(pack_clamp_s8(value));
#endif
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint unpackTestValue = unpackTestBuffer[0];

    uint index = 0;

    /*
     * Unpack without sign extension.
     */
    uint32_t4 unpackedU32 = __unpack_u8u32(unpackTestValue);
    // 0xFF
    outputBuffer[index++] = uint(unpackedU32.x);
    // 0x83
    outputBuffer[index++] = uint(unpackedU32.y);
    // 0x7A
    outputBuffer[index++] = uint(unpackedU32.z);
    // 0xD3
    outputBuffer[index++] = uint(unpackedU32.w);

#if !defined(WGSL)
    uint16_t4 unpackedU16 = __unpack_u8u16(unpackTestValue);
    // 0xFF
    outputBuffer[index++] = uint(unpackedU16.x);
    // 0x83
    outputBuffer[index++] = uint(unpackedU16.y);
    // 0x7A
    outputBuffer[index++] = uint(unpackedU16.z);
    // 0xD3
    outputBuffer[index++] = uint(unpackedU16.w);
#else 
    outputBuffer[index++] = 0xFFU;
    outputBuffer[index++] = 0x83U;
    outputBuffer[index++] = 0x7AU;
    outputBuffer[index++] = 0xD3U;
#endif

    /*
     * Unpack with sign extension.
     */
    int32_t4 unpackedS32 = __unpack_s8s32(unpackTestValue);
    // 0xFFFFFFFF
    outputBuffer[index++] = uint(unpackedS32.x);
    // 0xFFFFFF83
    outputBuffer[index++] = uint(unpackedS32.y);
    // 0x7A
    outputBuffer[index++] = uint(unpackedS32.z);
    // 0xFFFFFFD3
    outputBuffer[index++] = uint(unpackedS32.w);

#if !defined(WGSL)
    int16_t4 unpackedS16 = __unpack_s8s16(unpackTestValue);
    // 0xFFFFFFFF
    outputBuffer[index++] = uint(unpackedS16.x);
    // 0xFFFFFF83
    outputBuffer[index++] = uint(unpackedS16.y);
    // 0x7A
    outputBuffer[index++] = uint(unpackedS16.z);
    // 0xFFFFFFD3
    outputBuffer[index++] = uint(unpackedS16.w);
#else
    outputBuffer[index++] = 0xFFFFFFFFU;
    outputBuffer[index++] = 0xFFFFFF83U;
    outputBuffer[index++] = 0x7AU;
    outputBuffer[index++] = 0xFFFFFFD3U;
#endif


    /*
     * Pack without clamping, dropping unused bits.
     */
    uint32_t4 packU32TestValues = packTestBuffer[0];
    int32_t4 packS32TestValues = packU32TestValues;

    // 0xD4236A3F
    outputBuffer[index++] = __pack_u8(packU32TestValues);
    outputBuffer[index++] = __pack_s8(packS32TestValues);

#if !defined(WGSL)
    uint16_t4 packU16TestValues = int16_t4(int16_t(packU32TestValues.x), int16_t(packU32TestValues.y), 
                                           int16_t(packU32TestValues.z), int16_t(packU32TestValues.w));
    int16_t4 packS16TestValues = packU16TestValues;
    outputBuffer[index++] = __pack_u8(packU16TestValues);
    outputBuffer[index++] = __pack_s8(packS16TestValues);
#else
    outputBuffer[index++] = 0xD4236A3F;
    outputBuffer[index++] = 0xD4236A3F;
#endif

    /*
     * Pack with unsigned clamping.
     */
    int32_t4 packClampU32TestValues = packClampUTestBuffer[0];
    // 0xFEFFFF05
    outputBuffer[index++] = __pack_clamp_u8(packClampU32TestValues);

#if !defined(WGSL)
    int16_t4 packClampU16TestValues = int16_t4(int16_t(packClampU32TestValues.x), int16_t(packClampU32TestValues.y),
                                               int16_t(packClampU32TestValues.z), int16_t(packClampU32TestValues.w));
    outputBuffer[index++] = __pack_clamp_u8(packClampU16TestValues);
#else
    outputBuffer[index++] = 0xFEFFFF05;
#endif

    /*
     * Pack with signed clamping
     */
    int32_t4 packClampS32TestValues = packClampSTestBuffer[0];
    // 0x81807FFF
    outputBuffer[index++] = __pack_clamp_s8(packClampS32TestValues);

#if !defined(WGSL)
    int16_t4 packClampS16TestValues = int16_t4(int16_t(packClampS32TestValues.x), int16_t(packClampS32TestValues.y),
                                               int16_t(packClampS32TestValues.z), int16_t(packClampS32TestValues.w));
    outputBuffer[index++] = __pack_clamp_s8(packClampS16TestValues);
#else
    outputBuffer[index++] = 0x81807FFF;
#endif
}

