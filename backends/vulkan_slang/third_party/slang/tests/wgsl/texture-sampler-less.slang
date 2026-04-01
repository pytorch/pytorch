// WGSL supports "textureSample()" only in fragment shader
//TEST:SIMPLE(filecheck=WGSL): -stage fragment -entry fragMain -target wgsl

// TODO: offset variants requires the offset value to be compile-time constant
// But the keyword, `constexpr`, doesn't seem to work for the combined-textures
#define TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET 0

//WGSL:@binding(0) {{.*}}t1D_f32v3{{.*}} : texture_1d<f32>;
//WGSL:@binding(1) {{.*}}t1D_f32v3{{.*}} : sampler;
//TEST_INPUT: TextureSampler1D(size=4, content = zero):name t1D_f32v3
Sampler1D<float3> t1D_f32v3;

//WGSL:@binding(2) {{.*}}t2D_f32v3{{.*}} : texture_2d<f32>;
//WGSL:@binding(3) {{.*}}t2D_f32v3{{.*}} : sampler;
//TEST_INPUT: TextureSampler2D(size=4, content = zero):name t2D_f32v3
Sampler2D<float3> t2D_f32v3;

//TEST_INPUT: TextureSampler3D(size=4, content = zero):name t3D_f32v3
Sampler3D<float3> t3D_f32v3;
//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name tCube_f32v3
SamplerCube<float3> tCube_f32v3;

//TEST_INPUT: TextureSampler1D(size=4, content = zero, arrayLength=2):name t1DArray_f32v3
Sampler1DArray<float3> t1DArray_f32v3;
//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v3
Sampler2DArray<float3> t2DArray_f32v3;
//TEST_INPUT: TextureSamplerCube(size=4, content = zero, arrayLength=2):name tCubeArray_f32v3
SamplerCubeArray<float3> tCubeArray_f32v3;

//TEST_INPUT: TextureSampler1D(size=4, content = zero):name t1D_f32v4
Sampler1D<float4> t1D_f32v4;
//TEST_INPUT: TextureSampler2D(size=4, content = zero):name t2D_f32v4
Sampler2D<float4> t2D_f32v4;
//TEST_INPUT: TextureSampler3D(size=4, content = zero):name t3D_f32v4
Sampler3D<float4> t3D_f32v4;
//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name tCube_f32v4
SamplerCube<float4> tCube_f32v4;

//TEST_INPUT: TextureSampler1D(size=4, content = zero, arrayLength=2):name t1DArray_f32v4
Sampler1DArray<float4> t1DArray_f32v4;
//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v4
Sampler2DArray<float4> t2DArray_f32v4;
//TEST_INPUT: TextureSamplerCube(size=4, content = zero, arrayLength=2):name tCubeArray_f32v4
SamplerCubeArray<float4> tCubeArray_f32v4;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias CombinedDepth2d = _Texture<
    T,
    __Shape2D,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias CombinedDepth2d_array = _Texture<
    T,
    __Shape2D,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias CombinedDepthcube = _Texture<
    T,
    __ShapeCube,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias CombinedDepthcube_array = _Texture<
    T,
    __ShapeCube,
    1, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    1, // isCombined
    format
>;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name d2D
CombinedDepth2d<float> d2D;
//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name dCube
CombinedDepthcube<float> dCube;
//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength=2):name d2DArray
CombinedDepth2d_array<float> d2DArray;
//TEST_INPUT: TextureSamplerCube(size=4, content = zero, arrayLength=2):name dCubeArray
CombinedDepthcube_array<float> dCubeArray;

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;


bool TEST_texture<T>(
    Sampler1D<T> t1D,
    Sampler2D<T> t2D,
    Sampler3D<T> t3D,
    SamplerCube<T> tCube,
    Sampler1DArray<T> t1DArray,
    Sampler2DArray<T> t2DArray,
    SamplerCubeArray<T> tCubeArray
) where T:ITexelElement,IArithmetic
{
    // WGSL-LABEL: TEST_texture
    typealias Tvn = T;
    typealias Tv4 = vector<T.Element,4>;

    float u = 0;
    float u2 = 0.5;
    constexpr const float ddx = 0.0f;
    constexpr const float ddy = 0.0f;

    bool result = true
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
        && all(Tvn(T.Element(0)) == t1D.Sample(u))

        // WGSL: textureSample({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.Sample(float2(u, u)))

        // WGSL: textureSample({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.Sample(float3(u, u, u)))

        // WGSL: textureSample({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.Sample(normalize(float3(u, 1 - u, u))))

        // WGSL doesn't support textureSample for 1d_array and 3d_array; only 2d and cube

        // WGSL: textureSample({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.Sample(float3(u, u, 0)))

        // WGSL: textureSample({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.Sample(float4(normalize(float3(u, 1 - u, u)), 0)))

#if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
        // Offset variant

        // WGSL: textureSample({{\(*}}t1D
        && all(Tvn(T.Element(0)) == t1D.Sample(u, 1))

        // WGSL: textureSample({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.Sample(float2(u, u), int2(1, 1)))

        // WGSL: textureSample({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.Sample(float3(u, u, u), int3(1, 1, 1)))

        // WGSL doesn't support offset variant for cube and cube_array

        // WGSL: textureSample({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.Sample(float3(u, u, 0), int2(1, 1)))
#endif // #if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET

        // ===============
        // T SampleBias()
        // https://www.w3.org/TR/WGSL/#texturesamplebias
        // ===============

        // WGSL doesn't support Bias for 1D texture

        // WGSL: textureSampleBias({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleBias(float2(u, u), float(-1)))

        // WGSL: textureSampleBias({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleBias(float3(u, u, u), float(-1)))

        // WGSL: textureSampleBias({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.SampleBias(normalize(float3(u, 1 - u, u)), float(-1)))

        // WGSL: textureSampleBias({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleBias(float3(u, u, 0), float(-1)))

        // WGSL: textureSampleBias({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.SampleBias(float4(normalize(float3(u, 1 - u, u)), 0), float(-1)))

#if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
        // Offset variant

        // W-GSL: textureSampleBias({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleBias(float2(u, u), float(-1), int2(1, 1)))

        // W-GSL: textureSampleBias({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleBias(float3(u, u, u), float(-1), int3(1, 1, 1)))

        // W-GSL: textureSampleBias({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleBias(float3(u, u, 0), float(-1), int2(1, 1)))
#endif // #if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET

        // ===================================
        // T SampleLevel()
        // https://www.w3.org/TR/WGSL/#texturesamplelevel
        // ===================================

        // WGSL doesn't support textureSampleLevel for 1D texture

        // WGSL: textureSampleLevel({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleLevel(float2(u, u), 0))

        // WGSL: textureSampleLevel({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleLevel(float3(u, u, u), 0))

        // WGSL: textureSampleLevel({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.SampleLevel(normalize(float3(u, 1 - u, u)), 0))

        // WGSL: textureSampleLevel({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleLevel(float3(u, u, 0), 0))

        // WGSL: textureSampleLevel({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.SampleLevel(float4(normalize(float3(u, 1 - u, u)), 0), 0))

#if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
        // Offset variant

        // W-GSL: textureSampleLevel({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleLevel(float2(u, u), 0, int2(1, 1)))

        // W-GSL: textureSampleLevel({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleLevel(float3(u, u, u), 0, int3(1, 1, 1)))

        // W-GSL: textureSampleLevel({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleLevel(float3(u, u, 0), 0, int2(1, 1)))
#endif // #if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET

        // ==================
        // float SampleCmp()
        // https://www.w3.org/TR/WGSL/#texturesamplecompare
        // ==================

        // WGSL: textureSampleCompare({{\(*}}d2D
        && float(0) == d2D.SampleCmp(float2(u, u), 0)

        // WGSL: textureSampleCompare({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmp(float3(u, u, 0), 0)

        // WGSL: textureSampleCompare({{\(*}}dCube
        && float(0) == dCube.SampleCmp(normalize(float3(u, 1 - u, u)), 0)

        // WGSL: textureSampleCompare({{\(*}}dCubeArray
        && float(0) == dCubeArray.SampleCmp(float4(normalize(float3(u, 1 - u, u)), 0), 0)

#if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
        // Offset variant

        // W-GSL: textureSampleCompare({{\(*}}d2D
        && float(0) == d2D.SampleCmp(float2(u2, u), 0, int2(0, 0))

        // W-GSL: textureSampleCompare({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmp(float3(u2, u, u), 0, int2(0, 0))
#endif // #if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET

        // ===================================
        // float SampleCmpLevelZero()
        // https://www.w3.org/TR/WGSL/#texturesamplecomparelevel
        // ===================================

        // WGSL: textureSampleCompareLevel({{\(*}}d2D
        && float(0) == d2D.SampleCmpLevelZero(float2(u, u), 0)

        // WGSL: textureSampleCompareLevel({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmpLevelZero(float3(u, u, 0), 0)

        // WGSL: textureSampleCompareLevel({{\(*}}dCube
        && float(0) == dCube.SampleCmpLevelZero(normalize(float3(u, 1 - u, u)), 0)

        // WGSL: textureSampleCompareLevel({{\(*}}dCubeArray
        && float(0) == dCubeArray.SampleCmpLevelZero(float4(normalize(float3(u, 1-u, u)), 0), 0)

#if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
        // Offset variant

        // W-GSL: textureSampleCompareLevel({{\(*}}d2D
        && float(0) == d2D.SampleCmpLevelZero(float2(u2, u), 0, int2(0, 0))

        // W-GSL: textureSampleCompareLevel({{\(*}}d2DArray
        && float(0) == d2DArray.SampleCmpLevelZero(float3(u2, u, u), 0, int2(0, 0))
#endif // #if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET

        // ==================================
        // vector<T,4> Gather()
        // https://www.w3.org/TR/WGSL/#texturegather
        // ==================================

        // WGSL: textureGather({{.*}}t2D
        && all(Tv4(T.Element(0)) == t2D.Gather(float2(u, u)))

        // WGSL: textureGather({{.*}}tCube
        && all(Tv4(T.Element(0)) == tCube.Gather(normalize(float3(u, 1 - u, u))))

        // WGSL: textureGather({{.*}}t2DArray
        && all(Tv4(T.Element(0)) == t2DArray.Gather(float3(u, u, 0)))

        // WGSL: textureGather({{.*}}tCubeArray
        && all(Tv4(T.Element(0)) == tCubeArray.Gather(float4(normalize(float3(u, 1 - u, u)), 0)))

#if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
        // Offset variant

        // W-GSL: textureGather({{.*}}t2D
        && all(Tv4(T.Element(0)) == t2D.Gather(float2(u2, u), int2(0, 0)))

        // W-GSL: textureGather({{.*}}t2DArray
        && all(Tv4(T.Element(0)) == t2DArray.Gather(float3(u2, u, 0), int2(0, 0)))
#endif // #if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET

        // =====================================
        // T SampleGrad()
        // https://www.w3.org/TR/WGSL/#texturesamplegrad
        // =====================================

        // WGSL doesn't support textureSampleGrad for 1D textures

        // WGSL: textureSampleGrad({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleGrad(float2(u, u), float2(ddx, ddx), float2(ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleGrad(float3(u, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}tCube
        && all(Tvn(T.Element(0)) == tCube.SampleGrad(normalize(float3(u, 1 - u, u)), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleGrad(float3(u, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy)))

        // WGSL: textureSampleGrad({{\(*}}tCubeArray
        && all(Tvn(T.Element(0)) == tCubeArray.SampleGrad(float4(normalize(float3(u, 1 - u, u)), 0), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

#if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
        // Offset variant

        // W-GSL: textureSampleGrad({{\(*}}t2D
        && all(Tvn(T.Element(0)) == t2D.SampleGrad(float2(u2, u), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))

        // W-GSL: textureSampleGrad({{\(*}}t3D
        && all(Tvn(T.Element(0)) == t3D.SampleGrad(float3(u2, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy), int3(0, 0, 0)))

        // W-GSL: textureSampleGrad({{\(*}}t2DArray
        && all(Tvn(T.Element(0)) == t2DArray.SampleGrad(float3(u2, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))
#endif // #if TEST_WHEN_CONSTEXPR_WORKS_FOR_OFFSET
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
