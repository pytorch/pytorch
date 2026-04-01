//TEST:SIMPLE(filecheck=WGSL): -stage fragment -entry fragMain -target wgsl

// In WGSL, `textureSample` and other variants work only for f32 type.
// But textureLoad works for i32, u32 and f32.

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// f32 types

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_f32v3
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f32v3
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_f32v3
//TEST_INPUT: Texture2D(size=4, content = zero):name t2DMS_f32v3
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v3
Texture1D<float3>      t1D_f32v3;
Texture2D<float3>      t2D_f32v3;
Texture3D<float3>      t3D_f32v3;
Texture2DMS<float3>    t2DMS_f32v3;
Texture2DArray<float3> t2DArray_f32v3;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_f32v4
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f32v4
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_f32v4
//TEST_INPUT: Texture2D(size=4, content = zero):name t2DMS_f32v4
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v4
Texture1D<float4>      t1D_f32v4;
Texture2D<float4>      t2D_f32v4;
Texture3D<float4>      t3D_f32v4;
Texture2DMS<float4>    t2DMS_f32v4;
Texture2DArray<float4> t2DArray_f32v4;

// i32 types

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_i32v3
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_i32v3
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_i32v3
//TEST_INPUT: Texture2D(size=4, content = zero):name t2DMS_i32v3
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_i32v3
Texture1D<int3>      t1D_i32v3;
Texture2D<int3>      t2D_i32v3;
Texture3D<int3>      t3D_i32v3;
Texture2DMS<int3>    t2DMS_i32v3;
Texture2DArray<int3> t2DArray_i32v3;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_i32v4
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_i32v4
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_i32v4
//TEST_INPUT: Texture2D(size=4, content = zero):name t2DMS_i32v4
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_i32v4
Texture1D<int4>      t1D_i32v4;
Texture2D<int4>      t2D_i32v4;
Texture3D<int4>      t3D_i32v4;
Texture2DMS<int4>    t2DMS_i32v4;
Texture2DArray<int4> t2DArray_i32v4;

// u32 types

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_u32v3
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_u32v3
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_u32v3
//TEST_INPUT: Texture2D(size=4, content = zero):name t2DMS_u32v3
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_u32v3
Texture1D<uint3>      t1D_u32v3;
Texture2D<uint3>      t2D_u32v3;
Texture3D<uint3>      t3D_u32v3;
Texture2DMS<uint3>    t2DMS_u32v3;
Texture2DArray<uint3> t2DArray_u32v3;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_u32v4
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_u32v4
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_u32v4
//TEST_INPUT: Texture2D(size=4, content = zero):name t2DMS_u32v4
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_u32v4
Texture1D<uint4>      t1D_u32v4;
Texture2D<uint4>      t2D_u32v4;
Texture3D<uint4>      t3D_u32v4;
Texture2DMS<uint4>    t2DMS_u32v4;
Texture2DArray<uint4> t2DArray_u32v4;

// depth

__generic<let sampleCount:int=0, let format:int=0>
typealias Depth2D = _Texture<
    float,
    __Shape2D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<let sampleCount:int=0, let format:int=0>
typealias Depth2DMS = _Texture<
    float,
    __Shape2D,
    0, // isArray
    1, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<let sampleCount:int=0, let format:int=0>
typealias Depth2DArray = _Texture<
    float,
    __Shape2D,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

//TEST_INPUT: Texture2D(size=4, content = zero):name d2D
Depth2D d2D;
//TEST_INPUT: Texture2D(size=4, content = zero):name d2DMS
Depth2DMS d2DMS;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name d2DArray
Depth2DArray d2DArray;


bool TEST_textureLoad<T>(
    Texture1D<T> t1D,
    Texture2D<T> t2D,
    Texture3D<T> t3D,
    Texture2DMS<T> t2DMS,
    Texture2DArray<T> t2DArray)
    where T : ITexelElement, IArithmetic
{
    // WGSL-LABEL: TEST_textureLoad
    typealias Tvn = T;

    return true
        // ===================
        // T Load()
        // https://www.w3.org/TR/WGSL/#textureload
        // ===================

        // WGSL: textureLoad({{\(*}}t1D
        && all(Tvn(T.Element(0)) == t1D.Load(int2(0, 0)))

        // WGSL: textureLoad({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.Load(int3(0, 0, 0)))

        // WGSL: textureLoad({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.Load(int4(0, 0, 0, 0)))

        // WGSL: textureLoad({{\(*}}t2DMS
        && all(Tvn(T.Element(0)) == t2DMS.Load(int3(0, 0, 0)))

        // WGSL: textureLoad({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.Load(int4(0, 0, 0, 0)))
        ;
}

bool TEST_textureLoad_depth()
{
    // WGSL-LABEL: TEST_textureLoad_depth

    return true
        // ===================
        // T Load()
        // https://www.w3.org/TR/WGSL/#textureload
        // ===================

        // WGSL: textureLoad({{\(*}}d2D
        && all(0.f == d2D.Load(int3(0, 0, 0)))

        // WGSL: textureLoad({{\(*}}d2DMS
        && all(0.f == d2DMS.Load(int2(0, 0), 0))

        // WGSL: textureLoad({{\(*}}d2DArray
        && all(0.f == d2DArray.Load(int4(0, 0, 0, 0)))
        ;
}

void fragMain()
{
    bool result = true
        && TEST_textureLoad<float3>(t1D_f32v3, t2D_f32v3, t3D_f32v3, t2DMS_f32v3, t2DArray_f32v3)
        && TEST_textureLoad<float4>(t1D_f32v4, t2D_f32v4, t3D_f32v4, t2DMS_f32v4, t2DArray_f32v4)
        && TEST_textureLoad<int32_t3>(t1D_i32v3, t2D_i32v3, t3D_i32v3, t2DMS_i32v3, t2DArray_i32v3)
        && TEST_textureLoad<int32_t4>(t1D_i32v4, t2D_i32v4, t3D_i32v4, t2DMS_i32v4, t2DArray_i32v4)
        && TEST_textureLoad<uint32_t3>(t1D_u32v3, t2D_u32v3, t3D_u32v3, t2DMS_u32v3, t2DArray_u32v3)
        && TEST_textureLoad<uint32_t4>(t1D_u32v4, t2D_u32v4, t3D_u32v4, t2DMS_u32v4, t2DArray_u32v4)
        && TEST_textureLoad_depth()
        ;

    outputBuffer[0] = int(result);
}
