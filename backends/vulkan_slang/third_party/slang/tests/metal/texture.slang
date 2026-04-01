//TEST:SIMPLE(filecheck=METAL): -stage compute -entry computeMain -target metal
//TEST:SIMPLE(filecheck=METALLIB): -stage compute -entry computeMain -target metallib
//TEST:SIMPLE(filecheck=HLSL): -stage compute -entry computeMain -target hlsl -DEXCLUDE_INTEGER_TYPE
//TEST:SIMPLE(filecheck=SPIR): -stage compute -entry computeMain -target spirv -emit-spirv-directly -DEXCLUDE_HALF_TYPE -DEXCLUDE_SHORT_TYPE

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=FUNCTIONAL):-slang -compute -dx12 -profile cs_6_6 -use-dxil -shaderobj -output-using-type -xslang -DEXCLUDE_INTEGER_TYPE -xslang -DEXCLUDE_SM_6_7
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=FUNCTIONAL):-vk -emit-spirv-directly -compute -shaderobj -output-using-type -render-feature hardware-device -xslang -DEXCLUDE_HALF_TYPE -xslang -DEXCLUDE_SHORT_TYPE

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

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_f32
Texture1D<float4> t1D_f32;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f32
Texture2D<float4> t2D_f32;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_f32
Texture3D<float4> t3D_f32;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_f32
TextureCube<float4> tCube_f32;

//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_f32
Texture1DArray<float4> t1DArray_f32;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f32
Texture2DArray<float4> t2DArray_f32;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_f32
TextureCubeArray<float4> tCubeArray_f32;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_f16
Texture1D<half4> t1D_f16;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f16
Texture2D<half4> t2D_f16;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_f16
Texture3D<half4> t3D_f16;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_f16
TextureCube<half4> tCube_f16;

//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_f16
Texture1DArray<half4> t1DArray_f16;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f16
Texture2DArray<half4> t2DArray_f16;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_f16
TextureCubeArray<half4> tCubeArray_f16;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_i32
Texture1D<int4> t1D_i32;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_i32
Texture2D<int4> t2D_i32;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_i32
Texture3D<int4> t3D_i32;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_i32
TextureCube<int4> tCube_i32;

//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_i32
Texture1DArray<int4> t1DArray_i32;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_i32
Texture2DArray<int4> t2DArray_i32;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_i32
TextureCubeArray<int4> tCubeArray_i32;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_u32
Texture1D<uint4> t1D_u32;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_u32
Texture2D<uint4> t2D_u32;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_u32
Texture3D<uint4> t3D_u32;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_u32
TextureCube<uint4> tCube_u32;

//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_u32
Texture1DArray<uint4> t1DArray_u32;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_u32
Texture2DArray<uint4> t2DArray_u32;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_u32
TextureCubeArray<uint4> tCubeArray_u32;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_i16
Texture1D<int16_t4> t1D_i16;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_i16
Texture2D<int16_t4> t2D_i16;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_i16
Texture3D<int16_t4> t3D_i16;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_i16
TextureCube<int16_t4> tCube_i16;

//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_i16
Texture1DArray<int16_t4> t1DArray_i16;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_i16
Texture2DArray<int16_t4> t2DArray_i16;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_i16
TextureCubeArray<int16_t4> tCubeArray_i16;

//TEST_INPUT: Texture1D(size=4, content = zero):name t1D_u16
Texture1D<uint16_t4> t1D_u16;
//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_u16
Texture2D<uint16_t4> t2D_u16;
//TEST_INPUT: Texture3D(size=4, content = zero):name t3D_u16
Texture3D<uint16_t4> t3D_u16;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_u16
TextureCube<uint16_t4> tCube_u16;

//TEST_INPUT: Texture1D(size=4, content = zero, arrayLength=2):name t1DArray_u16
Texture1DArray<uint16_t4> t1DArray_u16;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_u16
Texture2DArray<uint16_t4> t2DArray_u16;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_u16
TextureCubeArray<uint16_t4> tCubeArray_u16;

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
) where T: ITexelElement, IArithmetic
{
    // METAL-LABEL: TEST_texture
    typealias Tvn = T;
    typealias Tv4 = vector<T.Element,4>;

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

    // METAL: .get_width(
    // METALLIB: call {{.*}}.get_width_texture_1d(
    t1D.GetDimensions(width);
    voidResult = voidResult && (uint(4) == width);

    // METAL: .get_width({{.*}}.get_num_mip_levels()
    // METALLIB: call {{.*}}.get_num_mip_levels_texture_1d(
    t1D.GetDimensions(0, width, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(3) == numLevels);

    // METAL: .get_width({{.*}}.get_height(
    // METALLIB: call {{.*}}.get_height_texture_2d(
    t2D.GetDimensions(width, height);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels()
    // METALLIB: call {{.*}}.get_num_mip_levels_texture_2d(
    t2D.GetDimensions(0, width, height, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(3) == numLevels);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_depth(
    // METALLIB: call {{.*}}.get_depth_texture_3d(
    t3D.GetDimensions(width, height, depth);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(4) == depth);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_depth({{.*}}.get_num_mip_levels()
    // METALLIB: call {{.*}}.get_num_mip_levels_texture_3d(
    t3D.GetDimensions(0, width, height, depth, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(4) == depth);
    voidResult = voidResult && (uint(3) == numLevels);

    // METAL: .get_width({{.*}}.get_height({{.*}}
    // METALLIB: call {{.*}}.get_height_texture_cube(
    tCube.GetDimensions(width, height);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels()
    // METALLIB: call {{.*}}.get_num_mip_levels_texture_cube(
    tCube.GetDimensions(0, width, height, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(3) == numLevels);

    // METAL: .get_width({{.*}}.get_array_size(
    // METALLIB: call {{.*}}.get_array_size_texture_1d_array(
    t1DArray.GetDimensions(width, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(2) == elements);

    // METAL: .get_width({{.*}}.get_num_mip_levels(
    // METALLIB: call {{.*}}.get_num_mip_levels_texture_1d_array(
    t1DArray.GetDimensions(0, width, elements, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(2) == elements);
    voidResult = voidResult && (uint(3) == numLevels);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_array_size(
    // METALLIB: call {{.*}}.get_array_size_texture_2d_array(
    t2DArray.GetDimensions(width, height, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels(
    // METALLIB: call {{.*}}.get_num_mip_levels_texture_2d_array(
    t2DArray.GetDimensions(0, width, height, elements, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);
    voidResult = voidResult && (uint(3) == numLevels);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_array_size(
    // METALLIB: call {{.*}}.get_array_size_texture_cube_array(
    tCubeArray.GetDimensions(width, height, elements);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);

    // METAL: .get_width({{.*}}.get_height({{.*}}.get_num_mip_levels(
    // METALLIB: call {{.*}}.get_num_mip_levels_texture_cube_array(
    tCubeArray.GetDimensions(0, width, height, elements, numLevels);
    voidResult = voidResult && (uint(4) == width);
    voidResult = voidResult && (uint(4) == height);
    voidResult = voidResult && (uint(2) == elements);
    voidResult = voidResult && (uint(3) == numLevels);

    bool result = voidResult
        // ===============================
        // float CalculateLevelOfDetail()
        // ===============================

        // Metal doesn't support LOD for 1D texture

        // METAL: t2D{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_2d(
        && float(0) == t2D.CalculateLevelOfDetail(samplerState, float2(u, u))

        // METAL: t3D{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_3d(
        && float(0) == t3D.CalculateLevelOfDetail(samplerState, float3(u, u, u))

        // METAL: tCube{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_cube(
        && float(0) == tCube.CalculateLevelOfDetail(samplerState, normalize(float3(u, 1 - u, u)))

        // METAL: t2DArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_2d_array(
        && float(0) == t2DArray.CalculateLevelOfDetail(samplerState, float2(u, u))

        // METAL: tCubeArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_cube_array(
        && float(0) == tCubeArray.CalculateLevelOfDetail(samplerState, normalize(float3(u, 1 - u, u)))

        // SamplerComparisonState variant

        // Functional tests that require SM higher than 6.6 (as of writing) cannot run.
#if !defined(EXCLUDE_SM_6_7)
        // METAL: t2D{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_2d(
        && float(0) == t2D.CalculateLevelOfDetail(shadowSampler, float2(u, u))

        // METAL: t3D{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_3d(
        && float(0) == t3D.CalculateLevelOfDetail(shadowSampler, float3(u, u, u))

        // METAL: tCube{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_cube(
        && float(0) == tCube.CalculateLevelOfDetail(shadowSampler, normalize(float3(u, 1 - u, u)))

        // METAL: t2DArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_2d_array(
        && float(0) == t2DArray.CalculateLevelOfDetail(shadowSampler, float2(u, u))

        // METAL: tCubeArray{{.*}}.calculate_clamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_clamped_lod_texture_cube_array(
        && float(0) == tCubeArray.CalculateLevelOfDetail(shadowSampler, normalize(float3(u, 1 - u, u)))
#endif

        // ========================================
        // float CalculateLevelOfDetailUnclamped()
        // ========================================

        // Metal doesn't support LOD for 1D texture

        // METAL: t2D{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_2d(
        && float(0) >= t2D.CalculateLevelOfDetailUnclamped(samplerState, float2(u, u))

        // METAL: t3D{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_3d(
        && float(0) >= t3D.CalculateLevelOfDetailUnclamped(samplerState, float3(u, u, u))

        // METAL: tCube{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_cube(
        && float(0) >= tCube.CalculateLevelOfDetailUnclamped(samplerState, normalize(float3(u, 1 - u, u)))

        // METAL: t2DArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_2d_array(
        && float(0) >= t2DArray.CalculateLevelOfDetailUnclamped(samplerState, float2(u, u))

        // METAL: tCubeArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_cube_array(
        && float(0) >= tCubeArray.CalculateLevelOfDetailUnclamped(samplerState, normalize(float3(u, 1 - u, u)))

        // SamplerComparisonState variant
#if !defined(EXCLUDE_SM_6_7)
        // METAL: t2D{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_2d(
        && float(0) >= t2D.CalculateLevelOfDetailUnclamped(shadowSampler, float2(u, u))

        // METAL: t3D{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_3d(
        && float(0) >= t3D.CalculateLevelOfDetailUnclamped(shadowSampler, float3(u, u, u))

        // METAL: tCube{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_cube(
        && float(0) >= tCube.CalculateLevelOfDetailUnclamped(shadowSampler, normalize(float3(u, 1 - u, u)))

        // METAL: t2DArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_2d_array(
        && float(0) >= t2DArray.CalculateLevelOfDetailUnclamped(shadowSampler, float2(u, u))

        // METAL: tCubeArray{{.*}}.calculate_unclamped_lod({{.*}}
        // METALLIB: call {{.*}}.calculate_unclamped_lod_texture_cube_array(
        && float(0) >= tCubeArray.CalculateLevelOfDetailUnclamped(shadowSampler, normalize(float3(u, 1 - u, u)))
#endif

        // ===========
        // T Sample()
        // ===========

        // METAL: t1D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_1d.v4f32(
        && all(Tvn(T.Element(0)) == t1D.Sample(samplerState, u))

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.Sample(samplerState, float2(u, u)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.Sample(samplerState, float3(u, u, u)))

        // METAL: tCube{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube.v4f32(
        && all(Tvn(T.Element(0)) == tCube.Sample(samplerState, normalize(float3(u, 1 - u, u))))

        // METAL: t1DArray{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_1d_array.v4f32(
        && all(Tvn(T.Element(0)) == t1DArray.Sample(samplerState, float2(u, 0)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.Sample(samplerState, float3(u, u, 0)))

        // METAL: tCubeArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube_array.v4f32(
        && all(Tvn(T.Element(0)) == tCubeArray.Sample(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // Offset variant

        // Metal doesn't support Offset for 1D and Cube texture

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.Sample(samplerState, float2(u, u), int2(1, 1)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.Sample(samplerState, float3(u, u, u), int3(1, 1, 1)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.Sample(samplerState, float3(u, u, 0), int2(1, 1)))

        // Clamp variant

        // Metal doesn't support Offset for 1D and Cube texture

        // METAL: t2D{{.*}}.sample({{.*}} min_lod_clamp(
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.Sample(samplerState, float2(u, u), int2(1, 1), float(1)))

        // METAL: t3D{{.*}}.sample({{.*}} min_lod_clamp(
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.Sample(samplerState, float3(u, u, u), int3(1, 1, 1), float(1)))

        // METAL: t2DArray{{.*}}.sample({{.*}} min_lod_clamp(
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.Sample(samplerState, float3(u, u, 0), int2(1, 1), float(1)))

        // ===============
        // T SampleBias()
        // ===============

        // Metal doesn't support Bias for 1D texture

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.SampleBias(samplerState, float2(u, u), float(-1)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.SampleBias(samplerState, float3(u, u, u), float(-1)))

        // METAL: tCube{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube.v4f32(
        && all(Tvn(T.Element(0)) == tCube.SampleBias(samplerState, normalize(float3(u, 1 - u, u)), float(-1)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.SampleBias(samplerState, float3(u, u, 0), float(-1)))

        // METAL: tCubeArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_cube_array.v4f32(
        && all(Tvn(T.Element(0)) == tCubeArray.SampleBias(samplerState, float4(normalize(float3(u, 1 - u, u)), 0), float(-1)))

        // Offset variant

        // Metal doesn't support Offset for 1D and Cube texture

        // METAL: t2D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.SampleBias(samplerState, float2(u, u), float(-1), int2(1, 1)))

        // METAL: t3D{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.SampleBias(samplerState, float3(u, u, u), float(-1), int3(1, 1, 1)))

        // METAL: t2DArray{{.*}}.sample({{.*}}
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.SampleBias(samplerState, float3(u, u, 0), float(-1), int2(1, 1)))

        // ===================================
        //  T SampleLevel()
        // ===================================

        // Metal doesn't support LOD for 1D texture

        // METAL: t2D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.SampleLevel(samplerState, float2(u, u), 0))

        // METAL: t3D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.SampleLevel(samplerState, float3(u, u, u), 0))

        // METAL: tCube{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_cube.v4f32(
        && all(Tvn(T.Element(0)) == tCube.SampleLevel(samplerState, normalize(float3(u, 1 - u, u)), 0))

        // METAL: t2DArray{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0))

        // METAL: tCubeArray{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_cube_array.v4f32(
        && all(Tvn(T.Element(0)) == tCubeArray.SampleLevel(samplerState, float4(normalize(float3(u, 1 - u, u)), 0), 0))

        // Offset variant

        // Metal doesn't support LOD for 1D texture

        // METAL: t2D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.SampleLevel(samplerState, float2(u, u), 0, int2(1, 1)))

        // METAL: t3D{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.SampleLevel(samplerState, float3(u, u, u), 0, int3(1, 1, 1)))

        // METAL: t2DArray{{.*}}.sample({{.*}} level(
        // METALLIB: call {{.*}}.sample_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.SampleLevel(samplerState, float3(u, u, 0), 0, int2(1, 1)))

        // ==================
        // float SampleCmp()
        // ==================

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(0) == d2D.SampleCmp(shadowSampler, float2(u, u), 0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d_array.f32(
        && float(0) == d2DArray.SampleCmp(shadowSampler, float3(u, u, 0), 0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube.f32(
        && float(0) == dCube.SampleCmp(shadowSampler, normalize(float3(u, 1 - u, u)), 0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube_array.f32(
        && float(0) == dCubeArray.SampleCmp(shadowSampler, float4(normalize(float3(u, 1 - u, u)), 0), 0)

        // Offset variant

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(0) == d2D.SampleCmp(shadowSampler, float2(u2, u), 0, int2(0, 0))

        // ===================================
        //  float SampleCmpLevelZero()
        // ===================================

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(0) == d2D.SampleCmpLevelZero(shadowSampler, float2(u, u), 0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d_array.f32(
        && float(0) == d2DArray.SampleCmpLevelZero(shadowSampler, float3(u, u, 0), 0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube.f32(
        && float(0) == dCube.SampleCmpLevelZero(shadowSampler, normalize(float3(u, 1 - u, u)), 0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube_array.f32(
        && float(0) == dCubeArray.SampleCmpLevelZero(shadowSampler, float4(normalize(float3(u, 1-u, u)), 0), 0)

        // Offset variant

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(0) == d2D.SampleCmpLevelZero(shadowSampler, float2(u2, u), 0, int2(0, 0))

        // ===================================
        //  float SampleCmpLevel()
        // ===================================

        // These require SM 6.7 for dx12 but functional compute tests currently do not run with the cs_6_7.
#if !defined(EXCLUDE_SM_6_7)
        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(0) == d2D.SampleCmpLevel(shadowSampler, float2(u, u), 0, 0.0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d_array.f32(
        && float(0) == d2DArray.SampleCmpLevel(shadowSampler, float3(u, u, 0), 0, 0.0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube.f32(
        && float(0) == dCube.SampleCmpLevel(shadowSampler, normalize(float3(u, 1 - u, u)), 0, 0.0)

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_cube_array.f32(
        && float(0) == dCubeArray.SampleCmpLevel(shadowSampler, float4(normalize(float3(u, 1-u, u)), 0), 0, 0.0)

        // Offset variant

        // METAL: {{.*}}.sample_compare(
        // METALLIB: call {{.*}}.sample_compare_depth_2d.f32(
        && float(0) == d2D.SampleCmpLevel(shadowSampler, float2(u2, u), 0, 0.0, int2(0, 0))
#endif

        // ==================================
        //  vector<T,4> Gather()
        // ==================================

        // METAL: t2D{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d.v4f32(
        && all(Tv4(T.Element(0)) == t2D.Gather(samplerState, float2(u, u)))

        // METAL: tCube{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_cube.v4f32(
        && all(Tv4(T.Element(0)) == tCube.Gather(samplerState, normalize(float3(u, 1 - u, u))))

        // METAL: t2DArray{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d_array.v4f32(
        && all(Tv4(T.Element(0)) == t2DArray.Gather(samplerState, float3(u, u, 0)))

        // METAL: tCubeArray{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_cube_array.v4f32(
        && all(Tv4(T.Element(0)) == tCubeArray.Gather(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // Offset variant

        // METAL: t2D{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d.v4f32(
        && all(Tv4(T.Element(0)) == t2D.Gather(samplerState, float2(u2, u), int2(0, 0)))

        // METAL: t2DArray{{.*}}.gather(
        // METALLIB: call {{.*}}.gather_texture_2d_array.v4f32(
        && all(Tv4(T.Element(0)) == t2DArray.Gather(samplerState, float3(u2, u, 0), int2(0, 0)))

        // =====================================
        //  T SampleGrad()
        // =====================================

        // Metal doesn't support LOD for 1D texture

        // METAL: t2D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_grad.v4f32(
        && all(Tvn(T.Element(0)) == t2D.SampleGrad(samplerState, float2(u, u), float2(ddx, ddx), float2(ddy, ddy)))

        // METAL: t3D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_3d_grad.v4f32(
        && all(Tvn(T.Element(0)) == t3D.SampleGrad(samplerState, float3(u, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // METAL: tCube{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_cube_grad.v4f32(
        && all(Tvn(T.Element(0)) == tCube.SampleGrad(samplerState, normalize(float3(u, 1 - u, u)), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy)))

        // METAL: t2DArray{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_array_grad.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.SampleGrad(samplerState, float3(u, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy)))

        // Offset variant

        // METAL: t2D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_grad.v4f32(
        && all(Tvn(T.Element(0)) == t2D.SampleGrad(samplerState, float2(u2, u), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))

        // METAL: t3D{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_3d_grad.v4f32(
        && all(Tvn(T.Element(0)) == t3D.SampleGrad(samplerState, float3(u2, u, u), float3(ddx, ddx, ddx), float3(ddy, ddy, ddy), int3(0, 0, 0)))

        // METAL: t2DArray{{.*}}.sample(
        // METALLIB: call {{.*}}.sample_texture_2d_array_grad.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.SampleGrad(samplerState, float3(u2, u, 0.0f), float2(ddx, ddx), float2(ddy, ddy), int2(0, 0)))

        // ===================
        //  T Load()
        // ===================

        // METAL: t1D{{.*}}.read(
        // METALLIB: call {{.*}}.read_texture_1d.v4f32(
        && all(Tvn(T.Element(0)) == t1D.Load(int2(0, 0)))

        // METAL: t2D{{.*}}.read(
        // METALLIB: call {{.*}}.read_texture_2d.v4f32(
        && all(Tvn(T.Element(0)) == t2D.Load(int3(0, 0, 0)))

        // METAL: t3D{{.*}}.read(
        // METALLIB: call {{.*}}.read_texture_3d.v4f32(
        && all(Tvn(T.Element(0)) == t3D.Load(int4(0, 0, 0, 0)))

        // METAL: t1DArray{{.*}}.read(
        // METALLIB: call {{.*}}.read_texture_1d_array.v4f32(
        && all(Tvn(T.Element(0)) == t1DArray.Load(int3(0, 0, 0)))

        // METAL: t2DArray{{.*}}.read(
        // METALLIB: call {{.*}}.read_texture_2d_array.v4f32(
        && all(Tvn(T.Element(0)) == t2DArray.Load(int4(0, 0, 0, 0)))

        // Offset variant

        // Metal doesn't support offset variants for Load
        ;

    return result;
}

[numthreads(2, 2, 1)]
void computeMain()
{
    // SPIR: OpEntryPoint
    // HLSL: void computeMain(

    bool result = true
        && TEST_texture<float3>(
            t1D_f32v3,
            t2D_f32v3,
            t3D_f32v3,
            tCube_f32v3,
            t1DArray_f32v3,
            t2DArray_f32v3,
            tCubeArray_f32v3)
        // Metal textures support `Tv` types, which "denotes a 4-component vector
        // type based on the templated type <T> for declaring the texture type:
        //  - If T is float, Tv is float4.
        //  - If T is half, Tv is half4.
        //  - If T is int, Tv is int4.
        //  - If T is uint, Tv is uint4.
        //  - If T is short, Tv is short4.
        //  - If T is ushort, Tv is ushort4."
        && TEST_texture<float4>(
            t1D_f32,
            t2D_f32,
            t3D_f32,
            tCube_f32,
            t1DArray_f32,
            t2DArray_f32,
            tCubeArray_f32)
#if !defined(EXCLUDE_HALF_TYPE)
        && TEST_texture<half4>(
            t1D_f16,
            t2D_f16,
            t3D_f16,
            tCube_f16,
            t1DArray_f16,
            t2DArray_f16,
            tCubeArray_f16)
#endif
#if !defined(EXCLUDE_INTEGER_TYPE)
        && TEST_texture<int4>(
            t1D_i32,
            t2D_i32,
            t3D_i32,
            tCube_i32,
            t1DArray_i32,
            t2DArray_i32,
            tCubeArray_i32)
        && TEST_texture<uint4>(
            t1D_u32,
            t2D_u32,
            t3D_u32,
            tCube_u32,
            t1DArray_u32,
            t2DArray_u32,
            tCubeArray_u32)
#if !defined(EXCLUDE_SHORT_TYPE)
        && TEST_texture<int16_t4>(
            t1D_i16,
            t2D_i16,
            t3D_i16,
            tCube_i16,
            t1DArray_i16,
            t2DArray_i16,
            tCubeArray_i16)
        && TEST_texture<uint16_t4>(
            t1D_u16,
            t2D_u16,
            t3D_u16,
            tCube_u16,
            t1DArray_u16,
            t2DArray_u16,
            tCubeArray_u16)
#endif
#endif
        ;

    // FUNCTIONAL: 1
    outputBuffer[0] = int(result);
}
