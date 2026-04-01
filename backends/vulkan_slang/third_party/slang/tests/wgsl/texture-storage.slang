//TEST:SIMPLE(filecheck=WGSL): -stage fragment -entry fragMain -target wgsl

// In WGSL, `textureSample` and other variants work only for f32 type.
// But textureLoad works for i32, u32 and f32.

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// f32 types

//TEST_INPUT: RWTexture1D(format=RG32Float, size=4, content = zero):name w1D_f32v2
//TEST_INPUT: RWTexture2D(format=RG32Float, size=4, content = zero):name w2D_f32v2
//TEST_INPUT: RWTexture3D(format=RG32Float, size=4, content = zero):name w3D_f32v2
//TEST_INPUT: RWTexture2D(format=RG32Float, size=4, content = zero, arrayLength=2):name w2DArray_f32v2
// WGSL: var w1D_f32v2{{[^:]*}}: texture_storage_1d<rg32float, read_write>
[format("rg32f")] RWTexture1D<float2>      w1D_f32v2;
[format("rg32f")] RWTexture2D<float2>      w2D_f32v2;
[format("rg32f")] RWTexture3D<float2>      w3D_f32v2;
[format("rg32f")] RWTexture2DArray<float2> w2DArray_f32v2;

//TEST_INPUT: RWTexture1D(format=RGBA8Unorm, size=4, content = zero):name w1D_f32v4
//TEST_INPUT: RWTexture2D(format=RGBA8Unorm, size=4, content = zero):name w2D_f32v4
//TEST_INPUT: RWTexture3D(format=RGBA8Unorm, size=4, content = zero):name w3D_f32v4
//TEST_INPUT: RWTexture2D(format=RGBA8Unorm, size=4, content = zero, arrayLength=2):name w2DArray_f32v4
// WGSL: var w1D_f32v4{{[^:]*}}: texture_storage_1d<rgba8unorm, read_write>
[format("rgba8")] RWTexture1D<float4>      w1D_f32v4;
[format("rgba8")] RWTexture2D<float4>      w2D_f32v4;
[format("rgba8")] RWTexture3D<float4>      w3D_f32v4;
[format("rgba8")] RWTexture2DArray<float4> w2DArray_f32v4;

//TEST_INPUT: RWTexture1D(format=BGRA8Unorm, size=4, content = zero):name w1D_f32_bgra_v4
//TEST_INPUT: RWTexture2D(format=BGRA8Unorm, size=4, content = zero):name w2D_f32_bgra_v4
//TEST_INPUT: RWTexture3D(format=BGRA8Unorm, size=4, content = zero):name w3D_f32_bgra_v4
//TEST_INPUT: RWTexture2D(format=BGRA8Unorm, size=4, content = zero, arrayLength=2):name w2DArray_f32_bgra_v4
// WGSL: var w1D_f32_bgra_v4{{[^:]*}}: texture_storage_1d<bgra8unorm, read_write>
[format("bgra8")] RWTexture1D<float4>      w1D_f32_bgra_v4;
[format("bgra8")] RWTexture2D<float4>      w2D_f32_bgra_v4;
[format("bgra8")] RWTexture3D<float4>      w3D_f32_bgra_v4;
[format("bgra8")] RWTexture2DArray<float4> w2DArray_f32_bgra_v4;

// i32 types

//TEST_INPUT: RWTexture1D(format=RG32Sint, size=4, content = zero):name w1D_i32v2
//TEST_INPUT: RWTexture2D(format=RG32Sint, size=4, content = zero):name w2D_i32v2
//TEST_INPUT: RWTexture3D(format=RG32Sint, size=4, content = zero):name w3D_i32v2
//TEST_INPUT: RWTexture2D(format=RG32Sint, size=4, content = zero, arrayLength=2):name w2DArray_i32v2
// WGSL: var w1D_i32v2{{[^:]*}}: texture_storage_1d<rg32sint, read_write>
[format("rg32i")] RWTexture1D<int2>      w1D_i32v2;
[format("rg32i")] RWTexture2D<int2>      w2D_i32v2;
[format("rg32i")] RWTexture3D<int2>      w3D_i32v2;
[format("rg32i")] RWTexture2DArray<int2> w2DArray_i32v2;

//TEST_INPUT: RWTexture1D(format=RGBA32Sint, size=4, content = zero):name w1D_i32v4
//TEST_INPUT: RWTexture2D(format=RGBA32Sint, size=4, content = zero):name w2D_i32v4
//TEST_INPUT: RWTexture3D(format=RGBA32Sint, size=4, content = zero):name w3D_i32v4
//TEST_INPUT: RWTexture2D(format=RGBA32Sint, size=4, content = zero, arrayLength=2):name w2DArray_i32v4
// WGSL: var w1D_i32v4{{[^:]*}}: texture_storage_1d<rgba32sint, read_write>
[format("rgba32i")] RWTexture1D<int4>      w1D_i32v4;
[format("rgba32i")] RWTexture2D<int4>      w2D_i32v4;
[format("rgba32i")] RWTexture3D<int4>      w3D_i32v4;
[format("rgba32i")] RWTexture2DArray<int4> w2DArray_i32v4;

// u32 types

//TEST_INPUT: RWTexture1D(format=RG32Uint, size=4, content = zero):name w1D_u32v2
//TEST_INPUT: RWTexture2D(format=RG32Uint, size=4, content = zero):name w2D_u32v2
//TEST_INPUT: RWTexture3D(format=RG32Uint, size=4, content = zero):name w3D_u32v2
//TEST_INPUT: RWTexture2D(format=RG32Uint, size=4, content = zero, arrayLength=2):name w2DArray_u32v2
// WGSL: var w1D_u32v2{{[^:]*}}: texture_storage_1d<rg32uint, read_write>
[format("rg32ui")] RWTexture1D<uint2>      w1D_u32v2;
[format("rg32ui")] RWTexture2D<uint2>      w2D_u32v2;
[format("rg32ui")] RWTexture3D<uint2>      w3D_u32v2;
[format("rg32ui")] RWTexture2DArray<uint2> w2DArray_u32v2;

//TEST_INPUT: RWTexture1D(format=RGBA16Uint, size=4, content = zero):name w1D_u32v4
//TEST_INPUT: RWTexture2D(format=RGBA16Uint, size=4, content = zero):name w2D_u32v4
//TEST_INPUT: RWTexture3D(format=RGBA16Uint, size=4, content = zero):name w3D_u32v4
//TEST_INPUT: RWTexture2D(format=RGBA16Uint, size=4, content = zero, arrayLength=2):name w2DArray_u32v4
// WGSL: var w1D_u32v4{{[^:]*}}: texture_storage_1d<rgba16uint, read_write>
[format("rgba16ui")] RWTexture1D<uint4>      w1D_u32v4;
[format("rgba16ui")] RWTexture2D<uint4>      w2D_u32v4;
[format("rgba16ui")] RWTexture3D<uint4>      w3D_u32v4;
[format("rgba16ui")] RWTexture2DArray<uint4> w2DArray_u32v4;


__generic<T, let sampleIndex:int, let format:int>
[ForceInline] // Workaround for a WGSL requirement that `texture_storage_Xd` must always be a global variable
bool TEST_textureStorage_StoreLoad(
    RWTexture1D<T, sampleIndex, format> w1D,
    RWTexture2D<T, sampleIndex, format> w2D,
    RWTexture3D<T, sampleIndex, format> w3D,
    RWTexture2DArray<T, sampleIndex, format> w2DArray)
    where T:ITexelElement, IArithmetic
{
    typealias Tvn = T;

    // ===================
    // o[i] = v;
    // https://www.w3.org/TR/WGSL/#texturestore
    // ===================

    // TODO: store before load
    // WGSL: textureStore({{\(*}}w1D
    w1D[0] = Tvn(T.Element(1));

    // WGSL: textureStore({{\(*}}w2D
    w2D[0] = Tvn(T.Element(1));

    // WGSL: textureStore({{\(*}}w3D
    w3D[0] = Tvn(T.Element(1));

    // WGSL: textureStore({{\(*}}w2DArray
    w2DArray[0] = Tvn(T.Element(1));

    return true
        // ===================
        // T Load()
        // https://www.w3.org/TR/WGSL/#textureload
        // ===================

        // WGSL: textureLoad({{\(*}}w1D
        && all(Tvn(T.Element(0)) == w1D.Load(0))

        // WGSL: textureLoad({{\(*}}w2D
        && all(Tvn(T.Element(0)) == w2D.Load(int2(0, 0)))

        // WGSL: textureLoad({{\(*}}w3D
        && all(Tvn(T.Element(0)) == w3D.Load(int3(0, 0, 0)))

        // WGSL: textureLoad({{\(*}}w2DArray
        && all(Tvn(T.Element(0)) == w2DArray.Load(int3(0, 0, 0)))
        ;
}

void fragMain()
{
    bool result = true
        && TEST_textureStorage_StoreLoad<float2>(w1D_f32v2, w2D_f32v2, w3D_f32v2, w2DArray_f32v2)
        && TEST_textureStorage_StoreLoad<float4>(w1D_f32v4, w2D_f32v4, w3D_f32v4, w2DArray_f32v4)
        && TEST_textureStorage_StoreLoad<float4>(w1D_f32_bgra_v4, w2D_f32_bgra_v4, w3D_f32_bgra_v4, w2DArray_f32_bgra_v4)
        && TEST_textureStorage_StoreLoad<int32_t2>(w1D_i32v2, w2D_i32v2, w3D_i32v2, w2DArray_i32v2)
        && TEST_textureStorage_StoreLoad<int32_t4>(w1D_i32v4, w2D_i32v4, w3D_i32v4, w2DArray_i32v4)
        && TEST_textureStorage_StoreLoad<uint32_t2>(w1D_u32v2, w2D_u32v2, w3D_u32v2, w2DArray_u32v2)
        && TEST_textureStorage_StoreLoad<uint32_t4>(w1D_u32v4, w2D_u32v4, w3D_u32v4, w2DArray_u32v4)
        ;

    outputBuffer[0] = int(result);
}
