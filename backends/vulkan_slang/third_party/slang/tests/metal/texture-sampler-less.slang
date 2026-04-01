//TEST:SIMPLE(filecheck=METAL): -stage compute -entry computeMain -target metal -DEMIT_SOURCE
//TEST:SIMPLE(filecheck=METALLIB): -stage compute -entry computeMain -target metallib -DEMIT_SOURCE -DMETALLIB
//TEST:SIMPLE(filecheck=HLSL): -stage compute -entry computeMain -target hlsl -DEMIT_SOURCE

// It appears that when kIROp_CombinedTextureSamplerGetSampler is used,
// "$1" refers to a sampler not the first parameter of the function.
// This causes to emit an invalid code when the implementation is shared
// between the combined and not-combined.
// The following code,
//   void GetDimensions(out float width) { __intrinsic_asm "*$1 = $0.get_width(0)"; }
// gets emitted as following,
//   (*(kernelContext_0->t1D_sampler_0) = (kernelContext_0->t1D_texture_0).get_width(0));
// when it is expected to be,
//   (*&width = (kernelContext_0->t1D_texture_0).get_width(0));
//#define TEST_WHEN_DEFAULT_SAMPLER_WORKS_PROPERLY

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// TODO: `SamplerXD` cannot be setup with TEST_INPUT method.

Sampler1D<float> t1D;
Sampler2D<float> t2D;
Sampler3D<float> t3D;
SamplerCube<float> tCube;

Sampler1DArray<float> t1DArray;
Sampler2DArray<float> t2DArray;
SamplerCubeArray<float> tCubeArray;

// Metal doc says "For depth texture types, T must be float."
__generic<T : ITexelElement, let sampleCount:int=0, let format:int=0>
typealias depth2d = _Texture<
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
typealias depth2d_array = _Texture<
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
typealias depthcube = _Texture<
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
typealias depthcube_array = _Texture<
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

depth2d<float> d2D;
depthcube<float> dCube;
depth2d_array<float> d2DArray;
depthcube_array<float> dCubeArray;

bool TEST_texture_float()
{
    // Metal textures support `Tv` types, which "denotes a 4-component vector
    // type based on the templated type <T> for declaring the texture type:
    //  - If T is float, Tv is float4.
    //  - If T is half, Tv is half4.
    //  - If T is int, Tv is int4.
    //  - If T is uint, Tv is uint4.
    //  - If T is short, Tv is short4.
    //  - If T is ushort, Tv is ushort4."
    typealias Tv = vector<float,4>;

    float u = 0;
    float u2 = 0.5;
    constexpr const float ddx = 0.0f;
    constexpr const float ddy = 0.0f;

    uint width = 0, height = 0, depth = 0;
    float fwidth = 0.0f, fheight = 0.0f, fdepth = 0.0f;
    uint numLevels = 0, elements = 0, sampleCount = 0;
    float fnumLevels = 0.0f, felements = 0.0f;

    bool voidResult = true;

    // ======================
    //  void GetDimensions()
    // ======================

#if defined(TEST_WHEN_DEFAULT_SAMPLER_WORKS_PROPERLY)
    // M-ETAL: .get_width(
    // M-ETALLIB: call {{.*}}.get_width_texture_1d(
    t1D.GetDimensions(width);
    voidResult = voidResult && (uint(4) == width);

    // M-ETAL: .get_width({{.*}}.get_num_mip_levels()
    // M-ETALLIB: call {{.*}}.get_num_mip_levels_texture_1d(
    t1D.GetDimensions(0, width, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(3) == numLevels);

    // M-ETAL: .get_width({{.*}}.get_height(
    // M-ETALLIB: call {{.*}}.get_height_texture_2d(
    t2D.GetDimensions(width, height);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels()
    // M-ETALLIB: call {{.*}}.get_num_mip_levels_texture_2d(
    t2D.GetDimensions(0, width, height, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(3) == numLevels);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_depth(
    // M-ETALLIB: call {{.*}}.get_depth_texture_3d(
    t3D.GetDimensions(width, height, depth);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(4) == depth);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_depth({{.*}}.get_num_mip_levels()
    // M-ETALLIB: call {{.*}}.get_num_mip_levels_texture_3d(
    t3D.GetDimensions(0, width, height, depth, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(4) == depth);
    voidResult = voidResult && (uint(3) == numLevels);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}
    // M-ETALLIB: call {{.*}}.get_height_texture_cube(
    tCube.GetDimensions(width, height);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels()
    // M-ETALLIB: call {{.*}}.get_num_mip_levels_texture_cube(
    tCube.GetDimensions(0, width, height, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(3) == numLevels);

    // M-ETAL: .get_width({{.*}}.get_array_size(
    // M-ETALLIB: call {{.*}}.get_array_size_texture_1d_array(
    t1DArray.GetDimensions(width, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(2) == elements);

    // M-ETAL: .get_width({{.*}}.get_num_mip_levels(
    // M-ETALLIB: call {{.*}}.get_num_mip_levels_texture_1d_array(
    t1DArray.GetDimensions(0, width, elements, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(2) == elements);
    voidResult = voidResult && (uint(3) == numLevels);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_array_size(
    // M-ETALLIB: call {{.*}}.get_array_size_texture_2d_array(
    t2DArray.GetDimensions(width, height, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels(
    // M-ETALLIB: call {{.*}}.get_num_mip_levels_texture_2d_array(
    t2DArray.GetDimensions(0, width, height, elements, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);
    voidResult = voidResult && (uint(3) == numLevels);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_array_size(
    // M-ETALLIB: call {{.*}}.get_array_size_texture_cube_array(
    tCubeArray.GetDimensions(width, height, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);

    // M-ETAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels(
    // M-ETALLIB: call {{.*}}.get_num_mip_levels_texture_cube_array(
    tCubeArray.GetDimensions(0, width, height, elements, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);
    voidResult = voidResult && (uint(3) == numLevels);
#endif

    bool result = voidResult
        // ===============================
        // float CalculateLevelOfDetail()
        // ===============================

        // METAL: t2D{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_2d(
        && float(0) == t2D.CalculateLevelOfDetail(float2(u, u))

        // METAL: t3D{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_3d(
        && float(0) == t3D.CalculateLevelOfDetail(float3(u, u, u))

        // METAL: tCube{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_cube(
        && float(0) == tCube.CalculateLevelOfDetail(normalize(float3(u, 1 - u, u)))

        // METAL: t2DArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_2d_array(
        && float(0) == t2DArray.CalculateLevelOfDetail(float2(u, u))

        // METAL: tCubeArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_cube_array(
        && float(0) == tCubeArray.CalculateLevelOfDetail(normalize(float3(u, 1 - u, u)))

        // Shadow variant

        // METAL: d2D{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_depth_2d(
        && float(0) == d2D.CalculateLevelOfDetail(float2(u, u))

        // METAL: dCube{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_depth_cube(
        && float(0) == dCube.CalculateLevelOfDetail(normalize(float3(u, 1 - u, u)))

        // METAL: d2DArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_depth_2d_array(
        && float(0) == d2DArray.CalculateLevelOfDetail(float2(u, u))

        // METAL: dCubeArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_depth_cube_array(
        && float(0) == dCubeArray.CalculateLevelOfDetail(normalize(float3(u, 1 - u, u)))

        // ========================================
        // float CalculateLevelOfDetailUnclamped()
        // ========================================

        // METAL: t2D{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_2d(
        && float(0) >= t2D.CalculateLevelOfDetailUnclamped(float2(u, u))

        // METAL: t3D{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_3d(
        && float(0) >= t3D.CalculateLevelOfDetailUnclamped(float3(u, u, u))

        // METAL: tCube{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_cube(
        && float(0) >= tCube.CalculateLevelOfDetailUnclamped(normalize(float3(u, 1 - u, u)))

        // METAL: t2DArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_2d_array(
        && float(0) >= t2DArray.CalculateLevelOfDetailUnclamped(float2(u, u))

        // METAL: tCubeArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_cube_array(
        && float(0) >= tCubeArray.CalculateLevelOfDetailUnclamped(normalize(float3(u, 1 - u, u)))

        // Shadow variant

        // METAL: d2D{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_depth_2d(
        && float(0) >= d2D.CalculateLevelOfDetailUnclamped(float2(u, u))

        // METAL: dCube{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_depth_cube(
        && float(0) >= dCube.CalculateLevelOfDetailUnclamped(normalize(float3(u, 1 - u, u)))

        // METAL: d2DArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_depth_2d_array(
        && float(0) >= d2DArray.CalculateLevelOfDetailUnclamped(float2(u, u))

        // METAL: dCubeArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_depth_cube_array(
        && float(0) >= dCubeArray.CalculateLevelOfDetailUnclamped(normalize(float3(u, 1 - u, u)))

        // ===========
        // T Sample()
        // ===========

        // METAL: t1D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_1d.v4f32(
        && all(Tv(1) == t1D.Sample(u))

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tv(1) == t2D.Sample(float2(u, u)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tv(1) == t3D.Sample(float3(u, u, u)))

        // METAL: tCube{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube.v4f32(
        && all(Tv(1) == tCube.Sample(normalize(float3(u, 1 - u, u))))

        // METAL: t1DArray{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_1d_array.v4f32(
        && all(Tv(1) == t1DArray.Sample(float2(u, 0)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.Sample(float3(u, u, 0)))

        // METAL: tCubeArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube_array.v4f32(
        && all(Tv(1) == tCubeArray.Sample(float4(normalize(float3(u, 1 - u, u)), 0)))

        // Offset variant

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tv(1) == t2D.Sample(float2(u, u), int2(1, 1)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tv(1) == t3D.Sample(float3(u, u, u), int3(1, 1, 1)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.Sample(float3(u, u, 0), int2(1, 1)))

        // Clamp variant

        // METAL: t2D{{.*}}.sample({{.*}} min_lod_clamp(
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tv(1) == t2D.Sample(float2(u, u), int2(1, 1), float(1)))

        // METAL: t3D{{.*}}.sample({{.*}} min_lod_clamp(
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tv(1) == t3D.Sample(float3(u, u, u), int3(1, 1, 1), float(1)))

        // METAL: t2DArray{{.*}}.sample({{.*}} min_lod_clamp(
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.Sample(float3(u, u, 0), int2(1, 1), float(1)))

        // ===============
        // T SampleBias()
        // ===============

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tv(1) == t2D.SampleBias(float2(u, u), float(-1)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tv(1) == t3D.SampleBias(float3(u, u, u), float(-1)))

        // METAL: tCube{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube.v4f32(
        && all(Tv(1) == tCube.SampleBias(normalize(float3(u, 1 - u, u)), float(-1)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.SampleBias(float3(u, u, 0), float(-1)))

        // METAL: tCubeArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube_array.v4f32(
        && all(Tv(1) == tCubeArray.SampleBias(float4(normalize(float3(u, 1 - u, u)), 0), float(-1)))

        // Offset variant

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tv(1) == t2D.SampleBias(float2(u, u), float(-1), int2(1, 1)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tv(1) == t3D.SampleBias(float3(u, u, u), float(-1), int3(1, 1, 1)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.SampleBias(float3(u, u, 0), float(-1), int2(1, 1)))

        // ===================================
        //  T SampleLevel()
        // ===================================

        // METAL: t2D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tv(1) == t2D.SampleLevel(float2(u, u), 0))

        // METAL: t3D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tv(1) == t3D.SampleLevel(float3(u, u, u), 0))

        // METAL: tCube{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_cube.v4f32(
        && all(Tv(1) == tCube.SampleLevel(normalize(float3(u, 1 - u, u)), 0))

        // METAL: t2DArray{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.SampleLevel(float3(u, u, 0), 0))

        // METAL: tCubeArray{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_cube_array.v4f32(
        && all(Tv(1) == tCubeArray.SampleLevel(float4(normalize(float3(u, 1 - u, u)), 0), 0))

        // Offset variant

        // METAL: t2D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tv(1) == t2D.SampleLevel(float2(u, u), 0, int2(1, 1)))

        // METAL: t3D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tv(1) == t3D.SampleLevel(float3(u, u, u), 0, int3(1, 1, 1)))

        // METAL: t2DArray{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.SampleLevel(float3(u, u, 0), 0, int2(1, 1)))

        // ==================
        // float SampleCmp()
        // ==================

        // METAL: d2D{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(1) == d2D.SampleCmp(float2(u, u), 0)

        // METAL: d2DArray{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d_array.f32(
        && float(1) == d2DArray.SampleCmp(float3(u, u, 0), 0)

        // METAL: dCube{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube.f32(
        && float(1) == dCube.SampleCmp(normalize(float3(u, 1 - u, u)), 0)

        // METAL: dCubeArray{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube_array.f32(
        && float(1) == dCubeArray.SampleCmp(float4(normalize(float3(u, 1 - u, u)), 0), 0)

        // Offset variant

        // METAL: d2D{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(1) == d2D.SampleCmp(float2(u2, u), 0, int2(0, 0))

        // ===================================
        //  float SampleCmpLevelZero()
        // ===================================

        // METAL: d2D{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(1) == d2D.SampleCmpLevelZero(float2(u, u), 0)

        // METAL: d2DArray{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d_array.f32(
        && float(1) == d2DArray.SampleCmpLevelZero(float3(u, u, 0), 0)

        // METAL: dCube{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube.f32(
        && float(1) == dCube.SampleCmpLevelZero(normalize(float3(u, 1 - u, u)), 0)

        // METAL: dCubeArray{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube_array.f32(
        && float(1) == dCubeArray.SampleCmpLevelZero(float4(normalize(float3(u, 1-u, u)), 0), 0)

        // Offset variant

        // METAL: d2D{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(1) == d2D.SampleCmpLevelZero(float2(u2, u), 0, int2(0, 0))

        // ===================================
        //  float SampleCmpLevel()
        // ===================================

        // METAL: d2D{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(1) == d2D.SampleCmpLevel(float2(u, u), 0, 1.0)

        // METAL: d2DArray{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d_array.f32(
        && float(1) == d2DArray.SampleCmpLevel(float3(u, u, 0), 0, 1.0)

        // METAL: dCube{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube.f32(
        && float(1) == dCube.SampleCmpLevel(normalize(float3(u, 1 - u, u)), 0, 1.0)

        // METAL: dCubeArray{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube_array.f32(
        && float(1) == dCubeArray.SampleCmpLevel(float4(normalize(float3(u, 1-u, u)), 0), 0, 1.0)

        // Offset variant

        // METAL: d2D{{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(1) == d2D.SampleCmpLevel(float2(u2, u), 0, 1.0, int2(0, 0))


        // ==================================
        //  vector<T,4> Gather()
        // ==================================

        // METAL: t2D{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d.v4f32(
        && all(Tv(1) == t2D.Gather(float2(u, u)))

        // METAL: tCube{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_cube.v4f32(
        && all(Tv(1) == tCube.Gather(normalize(float3(u, 1 - u, u))))

        // METAL: t2DArray{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.Gather(float3(u, u, 0)))

        // METAL: tCubeArray{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_cube_array.v4f32(
        && all(Tv(1) == tCubeArray.Gather(float4(normalize(float3(u, 1 - u, u)), 0)))

        // Offset variant

        // METAL: t2D{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d.v4f32(
        && all(Tv(1) == t2D.Gather(float2(u2, u), int2(0, 0)))

        // METAL: t2DArray{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.Gather(float3(u2, u, 0), int2(0, 0)))

        // =====================================
        //  T SampleGrad()
        // =====================================

        // METAL: t2D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_grad.v4f32(
        && all(Tv(1) == t2D.SampleGrad(float2(u, u), float2(ddx, ddx), float2(ddy, ddy)))

        // METAL: t3D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_3d_grad.v4f32(
        && all(Tv(1) == t3D.SampleGrad(float3(u, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // METAL: tCube{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_cube_grad.v4f32(
        && all(Tv(1) == tCube.SampleGrad(normalize(float3(u, 1 - u, u)), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // METAL: t2DArray{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_array_grad.v4f32(
        && all(Tv(1) == t2DArray.SampleGrad(float3(u, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy)))

        // Offset variant

        // METAL: t2D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_grad.v4f32(
        && all(Tv(1) == t2D.SampleGrad(float2(u2, u), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))

        // METAL: t3D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_3d_grad.v4f32(
        && all(Tv(1) == t3D.SampleGrad(float3(u2, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy), int3(0, 0, 0)))

        // METAL: t2DArray{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_array_grad.v4f32(
        && all(Tv(1) == t2DArray.SampleGrad(float3(u2, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))

        // ===================
        //  T Load()
        // ===================

#if defined(TEST_WHEN_DEFAULT_SAMPLER_WORKS_PROPERLY)
        // M-ETAL: t1D{{.*}}.read(
        // M-ETALLIB: call {{.*}}.read_texture_1d.v4f32(
        && all(Tv(1) == t1D.Load(int2(0, 0)))

        // M-ETAL: t2D{{.*}}.read(
        // M-ETALLIB: call {{.*}}.read_texture_2d.v4f32(
        && all(Tv(1) == t2D.Load(int3(0, 0, 0)))

        // M-ETAL: t3D{{.*}}.read(
        // M-ETALLIB: call {{.*}}.read_texture_3d.v4f32(
        && all(Tv(1) == t3D.Load(int4(0, 0, 0, 0)))

        // M-ETAL: t1DArray{{.*}}.read(
        // M-ETALLIB: call {{.*}}.read_texture_1d_array.v4f32(
        && all(Tv(1) == t1DArray.Load(int3(0, 0, 0)))

        // M-ETAL: t2DArray{{.*}}.read(
        // M-ETALLIB: call {{.*}}.read_texture_2d_array.v4f32(
        && all(Tv(1) == t2DArray.Load(int4(0, 0, 0, 0)))
#endif
        ;

    return result;
}


[numthreads(1, 1, 1)]
void computeMain()
{
    // SPIR: OpEntryPoint
    // HLSL: void computeMain(

    bool result = true
        && TEST_texture_float()
        ;

    // FUNCTIONAL: 1
    outputBuffer[0] = int(result);
}
