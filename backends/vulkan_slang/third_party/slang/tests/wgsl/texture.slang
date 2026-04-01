// WGSL supports "textureSample()" only in fragment shader
//TEST:SIMPLE(filecheck=WGSL): -stage fragment -entry fragMain -target wgsl

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_f32v3
Texture1D<float3> t1D_f32v3;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f32v3
Texture2D<float3> t2D_f32v3;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_f32v3
Texture3D<float3> t3D_f32v3;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_f32v3
TextureCube<float3> tCube_f32v3;
//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_f32v3
Texture1DArray<float3> t1DArray_f32v3;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v3
Texture2DArray<float3> t2DArray_f32v3;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_f32v3
TextureCubeArray<float3> tCubeArray_f32v3;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_f32v4
Texture1D<float4> t1D_f32v4;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f32v4
Texture2D<float4> t2D_f32v4;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_f32v4
Texture3D<float4> t3D_f32v4;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_f32v4
TextureCube<float4> tCube_f32v4;

//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_f32v4
Texture1DArray<float4> t1DArray_f32v4;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v4
Texture2DArray<float4> t2DArray_f32v4;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_f32v4
TextureCubeArray<float4> tCubeArray_f32v4;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depth2d = _Texture<
    T,
    __Shape2D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depth2d_array = _Texture<
    T,
    __Shape2D,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depthcube = _Texture<
    T,
    __ShapeCube,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depthcube_array = _Texture<
    T,
    __ShapeCube,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

//TEST_INPUT: Texture2D(size=4, content = zero):name d2D
depth2d<float> d2D;
//TEST_INPUT: TextureCube(size=4, content = zero):name dCube
depthcube<float> dCube;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name d2DArray
depth2d_array<float> d2DArray;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name dCubeArray
depthcube_array<float> dCubeArray;

//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;
//TEST_INPUT: Sampler:name shadowSampler
SamplerComparisonState shadowSampler;

bool TEST_texture<T>(
    Texture1D<T> t1D,
    Texture2D<T> t2D,
    Texture3D<T> t3D,
    TextureCube<T> tCube,
    Texture1DArray<T> t1DArray,
    Texture2DArray<T> t2DArray,
    TextureCubeArray<T> tCubeArray
) where T:ITexelElement, IArithmetic
{
    // WGSL-LABEL: TEST_texture
    typealias Tvn = T;
    typealias Tv4 = vector<T.Element,4>;

    float u = 0;
    float u2 = 0.5;
    constexpr const float ddx = 0.0f;
    constexpr const float ddy = 0.0f;

    uint width = 0, height = 0, depth = 0;
    uint elements = 0;

    bool voidResult = true;

    // ======================
    //  void GetDimensions()
    // ======================

    // WGSL: textureDimensions({{\(*}}t1D
    t1D.GetDimensions(width);
    voidResult = voidResult && (uint(4) == width);

    // WGSL: textureDimensions({{\(*}}t2D
    t2D.GetDimensions(width, height);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);

    // WGSL: textureDimensions({{\(*}}t3D
    t3D.GetDimensions(width, height, depth);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(4) == depth);

    // WGSL: textureDimensions({{\(*}}tCube
    tCube.GetDimensions(width, height);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);

    // WGSL: textureDimensions({{\(*}}t1DArray
    t1DArray.GetDimensions(width, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(2) == elements);

    // WGSL: textureDimensions({{\(*}}t2DArray
    t2DArray.GetDimensions(width, height, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);

    // WGSL: textureDimensions({{\(*}}tCubeArray
    tCubeArray.GetDimensions(width, height, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);

    bool result = voidResult
        // ===============================
        // float CalculateLevelOfDetail()
        // ===============================
        // WGSL doesn't have a way to calculate mip-map level for the given coordinate

        // ========================================
        // float CalculateLevelOfDetailUnclamped()
        // ========================================
        // WGSL doesn't have a way to calculate mip-map level for the given coordinate

        // ===========
        // T Sample()
        // https://www.w3.org/TR/WGSL/#texturesample
        // ===========

        // WGSL: textureSample({{\(*}}t1D
        && all(Tvn(T.Element(0)) == t1D.Sample(samplerState, u))

        // WGSL: textureSample({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.Sample(samplerState, float2(u, u)))

        // WGSL: textureSample({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.Sample(samplerState, float3(u, u, u)))

        // WGSL: textureSample({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.Sample(samplerState, normalize(float3(u, 1 - u, u))))

        // WGSL doesn't support textureSample for 1d_array and 3d_array; only 2d and cube

        // WGSL: textureSample({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.Sample(samplerState, float3(u, u, 0)))

        // WGSL: textureSample({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.Sample(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // Offset variant

        // WGSL: textureSample({{\(*}}t1D
        && all(Tvn(T.Element(0)) == t1D.Sample(samplerState, u, 1))

        // WGSL: textureSample({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.Sample(samplerState, float2(u, u), int2(1, 1)))

        // WGSL: textureSample({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.Sample(samplerState, float3(u, u, u), int3(1, 1, 1)))

        // WGSL doesn't support offset variant for cube and cube_array

        // WGSL: textureSample({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.Sample(samplerState, float3(u, u, 0), int2(1, 1)))

        // Clamp variant
        // WGSL doesn't support clamp variants for `textureSample()`

        // ===============
        // T SampleBias()
        // https://www.w3.org/TR/WGSL/#texturesamplebias
        // ===============

        // WGSL doesn't support Bias for 1D texture

        // WGSL: textureSampleBias({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleBias(samplerState, float2(u, u), float(-1)))

        // WGSL: textureSampleBias({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleBias(samplerState, float3(u, u, u), float(-1)))

        // WGSL: textureSampleBias({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.SampleBias(samplerState, normalize(float3(u, 1 - u, u)), float(-1)))

        // WGSL: textureSampleBias({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleBias(samplerState, float3(u, u, 0), float(-1)))

        // WGSL: textureSampleBias({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.SampleBias(samplerState, float4(normalize(float3(u, 1 - u, u)), 0), float(-1)))

        // Offset variant

        // WGSL: textureSampleBias({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleBias(samplerState, float2(u, u), float(-1), int2(1, 1)))

        // WGSL: textureSampleBias({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleBias(samplerState, float3(u, u, u), float(-1), int3(1, 1, 1)))

        // WGSL: textureSampleBias({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleBias(samplerState, float3(u, u, 0), float(-1), int2(1, 1)))

        // ===================================
        // T SampleLevel()
        // https://www.w3.org/TR/WGSL/#texturesamplelevel
        // ===================================

        // WGSL doesn't support textureSampleLevel for 1D texture

        // WGSL: textureSampleLevel({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleLevel(samplerState, float2(u, u), 0))

        // WGSL: textureSampleLevel({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleLevel(samplerState, float3(u, u, u), 0))

        // WGSL: textureSampleLevel({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.SampleLevel(samplerState, normalize(float3(u, 1 - u, u)), 0))

        // WGSL: textureSampleLevel({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0))

        // WGSL: textureSampleLevel({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.SampleLevel(samplerState, float4(normalize(float3(u, 1 - u, u)), 0), 0))

        // Offset variant

        // WGSL: textureSampleLevel({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleLevel(samplerState, float2(u, u), 0, int2(1, 1)))

        // WGSL: textureSampleLevel({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleLevel(samplerState, float3(u, u, u), 0, int3(1, 1, 1)))

        // WGSL: textureSampleLevel({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0, int2(1, 1)))

        // ==================
        // float SampleCmp()
        // https://www.w3.org/TR/WGSL/#texturesamplecompare
        // ==================

        // WGSL: textureSampleCompare({{\(*}}d2D
        && float(0) == d2D.SampleCmp(shadowSampler, float2(u, u), 0)

        // WGSL: textureSampleCompare({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmp(shadowSampler, float3(u, u, 0), 0)

        // WGSL: textureSampleCompare({{\(*}}dCube
        && float(0) == dCube.SampleCmp(shadowSampler, normalize(float3(u, 1 - u, u)), 0)

        // WGSL: textureSampleCompare({{\(*}}dCubeArray
        && float(0) == dCubeArray.SampleCmp(shadowSampler, float4(normalize(float3(u, 1 - u, u)), 0), 0)

        // Offset variant

        // WGSL doesn't support the offset variant for cube and cube_array

        // WGSL: textureSampleCompare({{\(*}}d2D
        && float(0) == d2D.SampleCmp(shadowSampler, float2(u2, u), 0, int2(0, 0))

        // WGSL: textureSampleCompare({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmp(shadowSampler, float3(u2, u, u), 0, int2(0, 0))

        // ===================================
        // float SampleCmpLevelZero()
        // https://www.w3.org/TR/WGSL/#texturesamplecomparelevel
        // ===================================

        // WGSL: textureSampleCompareLevel({{\(*}}d2D
        && float(0) == d2D.SampleCmpLevelZero(shadowSampler, float2(u, u), 0)

        // WGSL: textureSampleCompareLevel({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmpLevelZero(shadowSampler, float3(u, u, 0), 0)

        // WGSL: textureSampleCompareLevel({{\(*}}dCube
        && float(0) == dCube.SampleCmpLevelZero(shadowSampler, normalize(float3(u, 1 - u, u)), 0)

        // WGSL: textureSampleCompareLevel({{\(*}}dCubeArray
        && float(0) == dCubeArray.SampleCmpLevelZero(shadowSampler, float4(normalize(float3(u, 1-u, u)), 0), 0)

        // Offset variant

        // WGSL: textureSampleCompareLevel({{\(*}}d2D
        && float(0) == d2D.SampleCmpLevelZero(shadowSampler, float2(u2, u), 0, int2(0, 0))

        // WGSL: textureSampleCompareLevel({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmpLevelZero(shadowSampler, float3(u2, u, u), 0, int2(0, 0))

        // =====================================
        // T SampleGrad()
        // https://www.w3.org/TR/WGSL/#texturesamplegrad
        // =====================================

        // WGSL doesn't support textureSampleGrad for 1D textures

        // WGSL: textureSampleGrad({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleGrad(samplerState, float2(u, u), float2(ddx, ddx), float2(ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleGrad(samplerState, float3(u, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.SampleGrad(samplerState, normalize(float3(u, 1 - u, u)), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleGrad(samplerState, float3(u, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.SampleGrad(samplerState, float4(normalize(float3(u, 1 - u, u)), 0), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // Offset variant

        // WGSL: textureSampleGrad({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleGrad(samplerState, float2(u2, u), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))

        // WGSL: textureSampleGrad({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleGrad(samplerState, float3(u2, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy), int3(0, 0, 0)))

        // WGSL: textureSampleGrad({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleGrad(samplerState, float3(u2, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))
        ;

    return result;
}

void fragMain()
{
    bool result = true
        && TEST_texture<float3>(
            t1D_f32v3,
            t2D_f32v3,
            t3D_f32v3,
            tCube_f32v3,
            t1DArray_f32v3,
            t2DArray_f32v3,
            tCubeArray_f32v3)
        && TEST_texture<float4>(
            t1D_f32v4,
            t2D_f32v4,
            t3D_f32v4,
            tCube_f32v4,
            t1DArray_f32v4,
            t2DArray_f32v4,
            tCubeArray_f32v4)
        ;

    outputBuffer[0] = int(result);
}
