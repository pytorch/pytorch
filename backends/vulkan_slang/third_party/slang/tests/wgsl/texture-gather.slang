//TEST:SIMPLE(filecheck=WGSL): -stage fragment -entry fragMain -target wgsl

// In WGSL, `textureSample` and other variants work only for f32 type.
// But textureGather works for i32, u32 and f32.

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// f32 types

//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f32v3
Texture2D<float3> t2D_f32v3;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_f32v3
TextureCube<float3> tCube_f32v3;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v3
Texture2DArray<float3> t2DArray_f32v3;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_f32v3
TextureCubeArray<float3> tCubeArray_f32v3;

//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_f32v4
Texture2D<float4> t2D_f32v4;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_f32v4
TextureCube<float4> tCube_f32v4;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_f32v4
Texture2DArray<float4> t2DArray_f32v4;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_f32v4
TextureCubeArray<float4> tCubeArray_f32v4;

// i32 types

//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_i32v3
Texture2D<int3> t2D_i32v3;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_i32v3
TextureCube<int3> tCube_i32v3;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_i32v3
Texture2DArray<int3> t2DArray_i32v3;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_i32v3
TextureCubeArray<int3> tCubeArray_i32v3;

//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_i32v4
Texture2D<int4> t2D_i32v4;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_i32v4
TextureCube<int4> tCube_i32v4;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_i32v4
Texture2DArray<int4> t2DArray_i32v4;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_i32v4
TextureCubeArray<int4> tCubeArray_i32v4;

// u32 types

//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_u32v3
Texture2D<uint3> t2D_u32v3;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_u32v3
TextureCube<uint3> tCube_u32v3;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_u32v3
Texture2DArray<uint3> t2DArray_u32v3;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_u32v3
TextureCubeArray<uint3> tCubeArray_u32v3;

//TEST_INPUT: Texture2D(size=4, content = zero):name t2D_u32v4
Texture2D<uint4> t2D_u32v4;
//TEST_INPUT: TextureCube(size=4, content = zero):name tCube_u32v4
TextureCube<uint4> tCube_u32v4;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name t2DArray_u32v4
Texture2DArray<uint4> t2DArray_u32v4;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name tCubeArray_u32v4
TextureCubeArray<uint4> tCubeArray_u32v4;

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

__generic<let sampleCount:int=0, let format:int=0>
typealias DepthCube = _Texture<
    float,
    __ShapeCube,
    0, // isArray
    0, // isMS
    sampleCount,
    0, // access
    1, // isShadow
    0, // isCombined
    format
>;

__generic<let sampleCount:int=0, let format:int=0>
typealias DepthCubeArray = _Texture<
    float,
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
Depth2D d2D;
//TEST_INPUT: TextureCube(size=4, content = zero):name dCube
DepthCube dCube;
//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength=2):name d2DArray
Depth2DArray d2DArray;
//TEST_INPUT: TextureCube(size=4, content = zero, arrayLength=2):name dCubeArray
DepthCubeArray dCubeArray;

//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;
//TEST_INPUT: Sampler:name depthSampler
SamplerComparisonState depthSampler;


bool TEST_textureGather<T>(
    Texture2D<T> t2D,
    TextureCube<T> tCube,
    Texture2DArray<T> t2DArray,
    TextureCubeArray<T> tCubeArray)
    where T:IArithmetic,ITexelElement
{
    // WGSL-LABEL: TEST_textureGather
    typealias Tv4 = vector<T.Element,4>;

    float u = 0;
    float u2 = 0.5;

    return true
        // ==================================
        // vector<T,4> Gather()
        // https://www.w3.org/TR/WGSL/#texturegather
        // ==================================

        // WGSL-COUNT-2: textureGather({{.*}}0{{.*}}t2D
        && all(Tv4(T.Element(0)) == t2D.Gather(samplerState, float2(u, u)))
        && all(Tv4(T.Element(0)) == t2D.Gather(samplerState, float2(u2, u), int2(0,0)))

        // WGSL: textureGather({{.*}}0{{.*}}tCube
        && all(Tv4(T.Element(0)) == tCube.Gather(samplerState, normalize(float3(u, 1 - u, u))))

        // WGSL-COUNT-2: textureGather({{.*}}0{{.*}}t2DArray
        && all(Tv4(T.Element(0)) == t2DArray.Gather(samplerState, float3(u, u, 0)))
        && all(Tv4(T.Element(0)) == t2DArray.Gather(samplerState, float3(u2, u, 0), int2(0,0)))

        // WGSL: textureGather({{.*}}0{{.*}}tCubeArray
        && all(Tv4(T.Element(0)) == tCubeArray.Gather(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // ==================================
        // vector<T,4> GatherRed()
        // ==================================

        // WGSL-COUNT-2: textureGather({{.*}}0{{.*}}t2D
        && all(Tv4(T.Element(0)) == t2D.GatherRed(samplerState, float2(u, u)))
        && all(Tv4(T.Element(0)) == t2D.GatherRed(samplerState, float2(u2, u), int2(0,0)))

        // WGSL: textureGather({{.*}}0{{.*}}tCube
        && all(Tv4(T.Element(0)) == tCube.GatherRed(samplerState, normalize(float3(u, 1 - u, u))))

        // WGSL-COUNT-2: textureGather({{.*}}0{{.*}}t2DArray
        && all(Tv4(T.Element(0)) == t2DArray.GatherRed(samplerState, float3(u, u, 0)))
        && all(Tv4(T.Element(0)) == t2DArray.GatherRed(samplerState, float3(u2, u, 0), int2(0,0)))

        // WGSL: textureGather({{.*}}0{{.*}}tCubeArray
        && all(Tv4(T.Element(0)) == tCubeArray.GatherRed(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // ==================================
        // vector<T,4> GatherGreen()
        // ==================================

        // WGSL-COUNT-2: textureGather({{.*}}1{{.*}}t2D
        && all(Tv4(T.Element(0)) == t2D.GatherGreen(samplerState, float2(u, u)))
        && all(Tv4(T.Element(0)) == t2D.GatherGreen(samplerState, float2(u2, u), int2(0,0)))

        // WGSL: textureGather({{.*}}1{{.*}}tCube
        && all(Tv4(T.Element(0)) == tCube.GatherGreen(samplerState, normalize(float3(u, 1 - u, u))))

        // WGSL-COUNT-2: textureGather({{.*}}1{{.*}}t2DArray
        && all(Tv4(T.Element(0)) == t2DArray.GatherGreen(samplerState, float3(u, u, 0)))
        && all(Tv4(T.Element(0)) == t2DArray.GatherGreen(samplerState, float3(u2, u, 0), int2(0,0)))

        // WGSL: textureGather({{.*}}1{{.*}}tCubeArray
        && all(Tv4(T.Element(0)) == tCubeArray.GatherGreen(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // ==================================
        // vector<T,4> GatherBlue()
        // ==================================

        // WGSL-COUNT-2: textureGather({{.*}}2{{.*}}t2D
        && all(Tv4(T.Element(0)) == t2D.GatherBlue(samplerState, float2(u, u)))
        && all(Tv4(T.Element(0)) == t2D.GatherBlue(samplerState, float2(u2, u), int2(0,0)))

        // WGSL: textureGather({{.*}}2{{.*}}tCube
        && all(Tv4(T.Element(0)) == tCube.GatherBlue(samplerState, normalize(float3(u, 1 - u, u))))

        // WGSL-COUNT-2: textureGather({{.*}}2{{.*}}t2DArray
        && all(Tv4(T.Element(0)) == t2DArray.GatherBlue(samplerState, float3(u, u, 0)))
        && all(Tv4(T.Element(0)) == t2DArray.GatherBlue(samplerState, float3(u2, u, 0), int2(0,0)))

        // WGSL: textureGather({{.*}}2{{.*}}tCubeArray
        && all(Tv4(T.Element(0)) == tCubeArray.GatherBlue(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // ==================================
        // vector<T,4> GatherAlpha()
        // ==================================

        // WGSL-COUNT-2: textureGather({{.*}}3{{.*}}t2D
        && all(Tv4(T.Element(0)) == t2D.GatherAlpha(samplerState, float2(u, u)))
        && all(Tv4(T.Element(0)) == t2D.GatherAlpha(samplerState, float2(u2, u), int2(0,0)))

        // WGSL: textureGather({{.*}}3{{.*}}tCube
        && all(Tv4(T.Element(0)) == tCube.GatherAlpha(samplerState, normalize(float3(u, 1 - u, u))))

        // WGSL-COUNT-2: textureGather({{.*}}3{{.*}}t2DArray
        && all(Tv4(T.Element(0)) == t2DArray.GatherAlpha(samplerState, float3(u, u, 0)))
        && all(Tv4(T.Element(0)) == t2DArray.GatherAlpha(samplerState, float3(u2, u, 0), int2(0,0)))

        // WGSL: textureGather({{.*}}3{{.*}}tCubeArray
        && all(Tv4(T.Element(0)) == tCubeArray.GatherAlpha(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))
        ;
}

bool TEST_textureGather_depth()
{
    // WGSL-LABEL: TEST_textureGather_depth
    float u = 0.f;
    float u2 = 0.5f;
    float depthCompare = 0.5f;

    return true
        // ==================================
        // vector<T,4> Gather()
        // https://www.w3.org/TR/WGSL/#texturegather
        // ==================================

        // WGSL-COUNT-10: textureGather({{\(*}}d2D
        && all(float4(0) == d2D.Gather     (samplerState, float2(u, u)))
        && all(float4(0) == d2D.GatherRed  (samplerState, float2(u, u)))
        && all(float4(0) == d2D.GatherGreen(samplerState, float2(u, u)))
        && all(float4(0) == d2D.GatherBlue (samplerState, float2(u, u)))
        && all(float4(0) == d2D.GatherAlpha(samplerState, float2(u, u)))
        && all(float4(0) == d2D.Gather     (samplerState, float2(u, u), int2(0,0)))
        && all(float4(0) == d2D.GatherRed  (samplerState, float2(u, u), int2(0,0)))
        && all(float4(0) == d2D.GatherGreen(samplerState, float2(u, u), int2(0,0)))
        && all(float4(0) == d2D.GatherBlue (samplerState, float2(u, u), int2(0,0)))
        && all(float4(0) == d2D.GatherAlpha(samplerState, float2(u, u), int2(0,0)))

        // WGSL-COUNT-5: textureGather({{\(*}}dCube
        && all(float4(0) == dCube.Gather     (samplerState, normalize(float3(u, 1 - u, u))))
        && all(float4(0) == dCube.GatherRed  (samplerState, normalize(float3(u, 1 - u, u))))
        && all(float4(0) == dCube.GatherGreen(samplerState, normalize(float3(u, 1 - u, u))))
        && all(float4(0) == dCube.GatherBlue (samplerState, normalize(float3(u, 1 - u, u))))
        && all(float4(0) == dCube.GatherAlpha(samplerState, normalize(float3(u, 1 - u, u))))

        // WGSL-COUNT-10: textureGather({{\(*}}d2DArray
        && all(float4(0) == d2DArray.Gather     (samplerState, float3(u, u, 0)))
        && all(float4(0) == d2DArray.GatherRed  (samplerState, float3(u, u, 0)))
        && all(float4(0) == d2DArray.GatherGreen(samplerState, float3(u, u, 0)))
        && all(float4(0) == d2DArray.GatherBlue (samplerState, float3(u, u, 0)))
        && all(float4(0) == d2DArray.GatherAlpha(samplerState, float3(u, u, 0)))
        && all(float4(0) == d2DArray.Gather     (samplerState, float3(u, u, 0), int2(0,0)))
        && all(float4(0) == d2DArray.GatherRed  (samplerState, float3(u, u, 0), int2(0,0)))
        && all(float4(0) == d2DArray.GatherGreen(samplerState, float3(u, u, 0), int2(0,0)))
        && all(float4(0) == d2DArray.GatherBlue (samplerState, float3(u, u, 0), int2(0,0)))
        && all(float4(0) == d2DArray.GatherAlpha(samplerState, float3(u, u, 0), int2(0,0)))

        // WGSL-COUNT-5: textureGather({{\(*}}dCubeArray
        && all(float4(0) == dCubeArray.Gather     (samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))
        && all(float4(0) == dCubeArray.GatherRed  (samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))
        && all(float4(0) == dCubeArray.GatherGreen(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))
        && all(float4(0) == dCubeArray.GatherBlue (samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))
        && all(float4(0) == dCubeArray.GatherAlpha(samplerState, float4(normalize(float3(u, 1 - u, u)), 0)))

        // ==================================
        // vector<T,4> GatherCmp()
        // https://www.w3.org/TR/WGSL/#texturegathercompare
        // ==================================

        // WGSL-COUNT-2: textureGatherCompare({{\(*}}d2D
        && all(float4(0) == d2D.GatherCmp(depthSampler, float2(u, u), depthCompare))
        && all(float4(0) == d2D.GatherCmp(depthSampler, float2(u, u), depthCompare, int2(0,0)))

        // WGSL: textureGatherCompare({{\(*}}dCube
        && all(float4(0) == dCube.GatherCmp(depthSampler, normalize(float3(u, 1 - u, u)), depthCompare))

        // WGSL-COUNT-2: textureGatherCompare({{\(*}}d2DArray
        && all(float4(0) == d2DArray.GatherCmp(depthSampler, float3(u, u, 0), depthCompare))
        && all(float4(0) == d2DArray.GatherCmp(depthSampler, float3(u, u, 0), depthCompare, int2(0,0)))

        // WGSL: textureGatherCompare({{\(*}}dCubeArray
        && all(float4(0) == dCubeArray.GatherCmp(depthSampler, float4(normalize(float3(u, 1 - u, u)), 0), depthCompare))
        ;
}


void fragMain()
{
    bool result = true
        && TEST_textureGather<float3>(t2D_f32v3, tCube_f32v3, t2DArray_f32v3, tCubeArray_f32v3)
        && TEST_textureGather<float4>(t2D_f32v4, tCube_f32v4, t2DArray_f32v4, tCubeArray_f32v4)
        && TEST_textureGather<int32_t3>(t2D_i32v3, tCube_i32v3, t2DArray_i32v3, tCubeArray_i32v3)
        && TEST_textureGather<int32_t4>(t2D_i32v4, tCube_i32v4, t2DArray_i32v4, tCubeArray_i32v4)
        && TEST_textureGather<uint32_t3>(t2D_u32v3, tCube_u32v3, t2DArray_u32v3, tCubeArray_u32v3)
        && TEST_textureGather<uint32_t4>(t2D_u32v4, tCube_u32v4, t2DArray_u32v4, tCubeArray_u32v4)
        && TEST_textureGather_depth()
        ;

    outputBuffer[0] = int(result);
}
