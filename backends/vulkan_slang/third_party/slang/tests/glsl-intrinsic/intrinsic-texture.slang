//TEST:SIMPLE(filecheck=HLSL): -allow-glsl -stage compute  -entry computeMain -target hlsl -DHLSL
//TEST:SIMPLE(filecheck=HLSL): -allow-glsl -stage fragment -entry fragMain -target hlsl -DHLSL
//TEST:SIMPLE(filecheck=GLSL): -allow-glsl -stage compute  -entry computeMain -target glsl -DGLSL
//TEST:SIMPLE(filecheck=GLSL): -allow-glsl -stage fragment -entry fragMain -target glsl -DGLSL
//TEST:SIMPLE(filecheck=SPIR): -allow-glsl -stage compute  -entry computeMain -target spirv -DSPIRV
//TEST:SIMPLE(filecheck=SPIR): -allow-glsl -stage fragment -entry fragMain -target spirv -DSPIRV
//TEST:SIMPLE(filecheck=CUDA): -allow-glsl -stage compute  -entry computeMain -target cuda -DCUDA
//TEST:SIMPLE(filecheck=CUDA): -allow-glsl -stage fragment -entry fragMain -target cuda -DCUDA

//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -output-using-type -xslang -DSPIRV

// TODO: This test revealed a few problems for a path from GLSL to HLSL
//T-EST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-dx12 -compute -entry computeMain -allow-glsl -use-dxil -output-using-type -profile cs_6_6 -xslang -DHLSL


//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    int result;
} outputBuffer;

// float

//TEST_INPUT: TextureSampler1D(size=4, content = zero):name uniform_sampler1DShadow
uniform sampler1DShadow uniform_sampler1DShadow;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_sampler2DShadow
uniform sampler2DShadow uniform_sampler2DShadow;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name uniform_samplerCubeShadow
uniform samplerCubeShadow uniform_samplerCubeShadow;


//TEST_INPUT: TextureSampler1D(size=4, content = zero, arrayLength=2):name uniform_sampler1DArrayShadow
uniform sampler1DArrayShadow uniform_sampler1DArrayShadow;

//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength=2):name uniform_sampler2DArrayShadow
uniform sampler2DArrayShadow uniform_sampler2DArrayShadow;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_sampler2DRectShadow
uniform sampler2DRectShadow uniform_sampler2DRectShadow;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero, arrayLength=2):name uniform_samplerCubeArrayShadow
uniform samplerCubeArrayShadow uniform_samplerCubeArrayShadow;


//TEST_INPUT: TextureSampler1D(size=4, content = zero):name uniform_sampler1D
uniform sampler1D uniform_sampler1D;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_sampler2D
uniform sampler2D uniform_sampler2D;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_sampler2DRect
uniform sampler2DRect uniform_sampler2DRect;

//TEST_INPUT: TextureSampler3D(size=4, content = zero):name uniform_sampler3D
uniform sampler3D uniform_sampler3D;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name uniform_samplerCube
uniform samplerCube uniform_samplerCube;

//TEST_INPUT: TextureSampler1D(size=4, content = zero, arrayLength=2):name uniform_sampler1DArray
uniform sampler1DArray uniform_sampler1DArray;

//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength=2):name uniform_sampler2DArray
uniform sampler2DArray uniform_sampler2DArray;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero, arrayLength=2):name uniform_samplerCubeArray
uniform samplerCubeArray uniform_samplerCubeArray;

//TEST_INPUT: TextureSampler1D(size=4, content = zero):name uniform_samplerBuffer
uniform samplerBuffer uniform_samplerBuffer;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_sampler2DMS
uniform sampler2DMS uniform_sampler2DMS;

//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength=2):name uniform_sampler2DMSArray
uniform sampler2DMSArray uniform_sampler2DMSArray;

// int

//TEST_INPUT: TextureSampler1D(size=4, content = zero):name uniform_isampler1D
uniform isampler1D uniform_isampler1D;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_isampler2D
uniform isampler2D uniform_isampler2D;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_isampler2DRect
uniform isampler2DRect uniform_isampler2DRect;

//TEST_INPUT: TextureSampler3D(size=4, content = zero):name uniform_isampler3D
uniform isampler3D uniform_isampler3D;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name uniform_isamplerCube
uniform isamplerCube uniform_isamplerCube;

//TEST_INPUT: TextureSampler1D(size=4, content = zero):name uniform_isampler1DArray
uniform isampler1DArray uniform_isampler1DArray;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_isampler2DArray
uniform isampler2DArray uniform_isampler2DArray;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name uniform_isamplerCubeArray
uniform isamplerCubeArray uniform_isamplerCubeArray;

//TEST_INPUT: TextureSampler1D(size=4, content = zero):name uniform_isamplerBuffer
uniform isamplerBuffer uniform_isamplerBuffer;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_isampler2DMS
uniform isampler2DMS uniform_isampler2DMS;

//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength = 2):name uniform_isampler2DMSArray
uniform isampler2DMSArray uniform_isampler2DMSArray;

// uint

//TEST_INPUT: TextureSampler1D(size=4, content = zero):name uniform_usampler1D
uniform usampler1D uniform_usampler1D;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_usampler2D
uniform usampler2D uniform_usampler2D;

//TEST_INPUT: TextureSampler2D(size=4, content = zero):name uniform_usampler2DRect
uniform usampler2DRect uniform_usampler2DRect;

//TEST_INPUT: TextureSampler3D(size=4, content = zero):name uniform_usampler3D
uniform usampler3D uniform_usampler3D;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero):name uniform_usamplerCube
uniform usamplerCube uniform_usamplerCube;

//TEST_INPUT: TextureSampler1D(size=4, content = zero, arrayLength = 2):name uniform_usampler1DArray
uniform usampler1DArray uniform_usampler1DArray;

//TEST_INPUT: TextureSampler2D(size=4, content = zero, arrayLength = 2):name uniform_usampler2DArray
uniform usampler2DArray uniform_usampler2DArray;

//TEST_INPUT: TextureSamplerCube(size=4, content = zero, arrayLength = 2):name uniform_usamplerCubeArray
uniform usamplerCubeArray uniform_usamplerCubeArray;

//TEST_INPUT: Texture1D(size=4, content = zero):name uniform_usamplerBuffer
uniform usamplerBuffer uniform_usamplerBuffer;

//TEST_INPUT: Texture2D(size=4, content = zero):name uniform_usampler2DMS
uniform usampler2DMS uniform_usampler2DMS;

//TEST_INPUT: Texture2D(size=4, content = zero, arrayLength = 2):name uniform_usampler2DMSArray
uniform usampler2DMSArray uniform_usampler2DMSArray;

__generic<T : ITexelElement>
bool textureFuncs( Sampler1D<T> gsampler1D
    , Sampler2D<T> gsampler2D
    , Sampler2DRect<T> gsampler2DRect
    , Sampler3D<T> gsampler3D
    , SamplerCube<T> gsamplerCube
    , Sampler1DArray<T> gsampler1DArray
    , Sampler2DArray<T> gsampler2DArray
    , SamplerCubeArray<T> gsamplerCubeArray
    , SamplerBuffer<T> gsamplerBuffer
    , Sampler2DMS<T> gsampler2DMS
    , Sampler2DMSArray<T> gsampler2DMSArray
    , bool ignoreResult
)
{
    // GLSL-LABEL: textureFuncs_0
    typealias gvec4 = vector<T.Element,4>;

    constexpr float coord = 0.5;

    constexpr ivec2 offset2D = ivec2(2);
    constexpr ivec3 offset3D = ivec3(3);
    constexpr ivec2 offsets[4] = { ivec2(1), ivec2(2), ivec2(3), ivec2(4) };

    bool ignoreResultF32 = ignoreResult && T.Element is float;
    bool ignoreResultI32 = ignoreResult && T.Element is int32_t;

    return true
        // 8.9.1. Texture Query Functions

        // GLSL: textureSize({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && int(4) == textureSize(gsampler1D, int(0))

        // GLSL: textureSize({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec2(4) == textureSize(gsampler2D, int(0))

        // GLSL: textureSize({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec3(4) == textureSize(gsampler3D, int(0))

        // GLSL: textureSize({{.*}}samplerCube
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec2(4) == textureSize(gsamplerCube, int(0))

        // GLSL: textureSize({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && int(4) == textureSize(uniform_sampler1DShadow, int(0))

        // GLSL: textureSize({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && (ivec2(4) == textureSize(uniform_sampler2DShadow, int(0)) || ignoreResult)

        // GLSL: textureSize({{.*}}samplerCubeShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && (ivec2(4) == textureSize(uniform_samplerCubeShadow, int(0)) || ignoreResult)

        // GLSL: textureSize({{.*}}samplerCubeArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && (ivec3(4,4,2) == textureSize(gsamplerCubeArray, int(0)) || ignoreResultI32)

        // GLSL: textureSize({{.*}}samplerCubeArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArrayShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec3(4,4,2) == textureSize(uniform_samplerCubeArrayShadow, int(0))

        // GLSL: textureSize({{.*}}sampler2DRect
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec2(4) == textureSize(gsampler2DRect)

        // GLSL: textureSize({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec2(4) == textureSize(uniform_sampler2DRectShadow)

        // GLSL: textureSize({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && (ivec2(4,2) == textureSize(gsampler1DArray, int(0)) || ignoreResultI32 )

        // GLSL: textureSize({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && (ivec2(4,2) == textureSize(uniform_sampler1DArrayShadow, int(0)) || ignoreResult)

        // GLSL: textureSize({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && (ivec3(4,4,2) == textureSize(gsampler2DArray, int(0)) || ignoreResultI32)

        // GLSL: textureSize({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec3(4,4,2) == textureSize(uniform_sampler2DArrayShadow, int(0))

        // GLSL: imageSize({{.*}}samplerBuffer
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerBuffer
        // SPIR: OpImageQuerySize{{.*}}[[LOAD]]
        && (int(4) == textureSize(gsamplerBuffer) || ignoreResult)

        // GLSL: textureSize({{.*}}sampler2DMS
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DMS
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec2(4) == textureSize(gsampler2DMS)

        // GLSL: textureSize({{.*}}sampler2DMSArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DMSArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySize{{.*}}[[IMAGE]]
        && ivec3(4,4,2) == textureSize(gsampler2DMSArray)

        // GLSL: textureQueryLod({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(gsampler1D, float(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(gsampler2D, vec2(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(gsampler3D, vec3(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}samplerCube
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(gsamplerCube, vec3(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(gsampler1DArray, float(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(gsampler2DArray, vec2(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}samplerCubeArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(gsamplerCubeArray, vec3(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(uniform_sampler1DShadow, float(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(uniform_sampler2DShadow, vec2(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}samplerCubeShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeShadow
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(uniform_samplerCubeShadow, vec3(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(uniform_sampler1DArrayShadow, float(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(uniform_sampler2DArrayShadow, vec2(coord)) * vec2(1,0))

        // GLSL: textureQueryLod({{.*}}samplerCubeArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArrayShadow
        // SPIR: OpImageQueryLod{{.*}}[[LOAD]]
        && vec2(0) == (textureQueryLod(uniform_samplerCubeArrayShadow, vec3(coord)) * vec2(1,0))

        // GLSL: textureQueryLevels({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(gsampler1D)

        // GLSL: textureQueryLevels({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(gsampler2D)

        // GLSL: textureQueryLevels({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(gsampler3D)

        // GLSL: textureQueryLevels({{.*}}samplerCube
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(gsamplerCube)

        // GLSL: textureQueryLevels({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(gsampler1DArray)

        // GLSL: textureQueryLevels({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(gsampler2DArray)

        // GLSL: textureQueryLevels({{.*}}samplerCubeArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(gsamplerCubeArray)

        // GLSL: textureQueryLevels({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(uniform_sampler1DShadow)

        // GLSL: textureQueryLevels({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(uniform_sampler2DShadow)

        // GLSL: textureQueryLevels({{.*}}samplerCubeShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(uniform_samplerCubeShadow)

        // GLSL: textureQueryLevels({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(uniform_sampler1DArrayShadow)

        // GLSL: textureQueryLevels({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(uniform_sampler2DArrayShadow)

        // GLSL: textureQueryLevels({{.*}}samplerCubeArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArrayShadow
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQueryLevels{{.*}}[[IMAGE]]
        && int(3) == textureQueryLevels(uniform_samplerCubeArrayShadow)

        // GLSL: textureSamples({{.*}}sampler2DMS
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DMS
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySamples{{.*}}[[IMAGE]]
        && int(1) == textureSamples(gsampler2DMS)

        // GLSL: textureSamples({{.*}}sampler2DMSArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DMSArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageQuerySamples{{.*}}[[IMAGE]]
        && int(1) == textureSamples(gsampler2DMSArray)

        // 8.9.2. Texel Lookup Functions

        // GLSL: texture({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsampler1D, float(coord))

        // GLSL: texture({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == texture(gsampler1D, float(coord), float(0))

        // GLSL: texture({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsampler2D, vec2(coord))

        // GLSL: texture({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == texture(gsampler2D, vec2(coord), float(0))

        // GLSL: texture({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsampler3D, vec3(coord))

        // GLSL: texture({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == texture(gsampler3D, vec3(coord), float(0))

        // GLSL: texture({{.*}}samplerCube
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsamplerCube, vec3(coord))

        // GLSL: texture({{.*}}samplerCube
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == texture(gsamplerCube, vec3(coord), float(0))

        // GLSL: texture({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]
        && float(0) == texture(uniform_sampler1DShadow, vec3(coord))

        // GLSL: texture({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && float(0) == texture(uniform_sampler1DShadow, vec3(coord), float(0))

        // GLSL: texture({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]
        && (float(0) == texture(uniform_sampler2DShadow, vec3(coord)) || ignoreResultF32)

        // GLSL: texture({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && (float(0) == texture(uniform_sampler2DShadow, vec3(coord), float(0)) || ignoreResultF32)

        // GLSL: texture({{.*}}samplerCubeShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]
        && (float(0) == texture(uniform_samplerCubeShadow, vec4(coord)) || ignoreResultF32)

        // GLSL: texture({{.*}}samplerCubeShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && (float(0) == texture(uniform_samplerCubeShadow, vec4(coord), float(0)) || ignoreResultF32)

        // GLSL: texture({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsampler2DArray, vec3(coord))

        // GLSL: texture({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == texture(gsampler2DArray, vec3(coord), float(0))

        // GLSL: texture({{.*}}samplerCubeArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsamplerCubeArray, vec4(coord))

        // GLSL: texture({{.*}}samplerCubeArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == texture(gsamplerCubeArray, vec4(coord), float(0))

        // GLSL: texture({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsampler1DArray, vec2(coord))

        // GLSL: texture({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == texture(gsampler1DArray, vec2(coord), float(0))

        // GLSL: texture({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]
        && float(0) == texture(uniform_sampler1DArrayShadow, vec3(coord))

        // GLSL: texture({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]{{.*}} Bias %
        && float(0) == texture(uniform_sampler1DArrayShadow, vec3(coord), float(0))

        // GLSL: texture({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]
        && float(0) == texture(uniform_sampler2DArrayShadow, vec4(coord))

        // GLSL: texture({{.*}}sampler2DRect
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleImplicitLod{{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == texture(gsampler2DRect, vec2(coord))

        // GLSL: texture({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]
        && float(0) == texture(uniform_sampler2DRectShadow, vec3(coord))

        // GLSL: texture({{.*}}samplerCubeArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArrayShadow
        // SPIR: OpImageSampleDrefImplicitLod{{.*}}[[LOAD]]
        && float(0) == texture(uniform_samplerCubeArrayShadow, vec4(coord), float(0))

        // GLSL: textureProj({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec2(coord))

        // GLSL: textureProj({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec2(coord), float(0))

        // GLSL: textureProj({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec4(coord))

        // GLSL: textureProj({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec4(coord), float(0))

        // GLSL: textureProj({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec3(coord))

        // GLSL: textureProj({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec3(coord), float(0))

        // GLSL: textureProj({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec4(coord))

        // GLSL: textureProj({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec4(coord), float(0))

        // GLSL: textureProj({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureProj(gsampler3D, vec4(coord))

        // GLSL: textureProj({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias %
        && gvec4(T.Element(0)) == textureProj(gsampler3D, vec4(coord), float(0))

        // GLSL: textureProj({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]
        && float(0) == textureProj(uniform_sampler1DShadow, vec4(coord))

        // GLSL: textureProj({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]{{.*}} Bias %
        && float(0) == textureProj(uniform_sampler1DShadow, vec4(coord), float(0))

        // GLSL: textureProj({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]
        && (float(0) == textureProj(uniform_sampler2DShadow, vec4(coord)) || ignoreResultF32)

        // GLSL: textureProj({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]{{.*}} Bias %
        && (float(0) == textureProj(uniform_sampler2DShadow, vec4(coord), float(0)) || ignoreResultF32)

        // GLSL-COUNT-2: textureProj({{.*}}sampler2DRect
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureProj(gsampler2DRect, vec3(coord))
        && gvec4(T.Element(0)) == textureProj(gsampler2DRect, vec4(coord))

        // GLSL: textureProj({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]
        && float(0) == textureProj(uniform_sampler2DRectShadow, vec4(coord))

        // GLSL: textureLod({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureLod(gsampler1D, float(coord), float(0))

        // GLSL: textureLod({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureLod(gsampler2D, vec2(coord), float(0))

        // GLSL: textureLod({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureLod(gsampler3D, vec3(coord), float(0))

        // GLSL: textureLod({{.*}}samplerCube
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureLod(gsamplerCube, vec3(coord), float(0))

        // GLSL: textureLod({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && (float(0) == textureLod(uniform_sampler2DShadow, vec3(coord), float(0)) || ignoreResultF32)

        // GLSL: textureLod({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && float(0) == textureLod(uniform_sampler1DShadow, vec3(coord), float(0))

        // GLSL: textureLod({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureLod(gsampler1DArray, vec2(coord), float(0))

        // GLSL: textureLod({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && float(0) == textureLod(uniform_sampler1DArrayShadow, vec3(coord), float(0))

        // GLSL: textureLod({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureLod(gsampler2DArray, vec3(coord), float(0))

        // GLSL: textureLod({{.*}}samplerCubeArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureLod(gsamplerCubeArray, vec4(coord), float(0))

        // GLSL: textureOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler1D, float(coord), int(1))

        // GLSL: textureOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler1D, float(coord), int(0), float(0))

        // GLSL: textureOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler2D, vec2(coord), offset2D)

        // GLSL: textureOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler2D, vec2(coord), offset2D, float(0))

        // GLSL: textureOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler3D, vec3(coord), offset3D)

        // GLSL: textureOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler3D, vec3(coord), offset3D, float(0))

        // GLSL: textureOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && (float(0) == textureOffset(uniform_sampler2DShadow, vec3(coord), offset2D) || ignoreResultF32)

        // GLSL: textureOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && (float(0) == textureOffset(uniform_sampler2DShadow, vec3(coord), offset2D, float(0)) || ignoreResultF32)

        // GLSL: textureOffset({{.*}}sampler2DRect
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler2DRect, vec2(coord), offset2D)

        // GLSL: textureOffset({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && float(0) == textureOffset(uniform_sampler2DRectShadow, vec3(coord), offset2D)

        // GLSL: textureOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && float(0) == textureOffset(uniform_sampler1DShadow, vec3(coord), int(0))

        // GLSL: textureOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && float(0) == textureOffset(uniform_sampler1DShadow, vec3(coord), int(0), float(0))

        // GLSL: textureOffset({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler1DArray, vec2(coord), int(1))

        // GLSL: textureOffset({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler1DArray, vec2(coord), int(0), float(0))

        // GLSL: textureOffset({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}}ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler2DArray, vec3(coord), offset2D)

        // GLSL: textureOffset({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureOffset(gsampler2DArray, vec3(coord), offset2D, float(0))

        // GLSL: textureOffset({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && float(0) == textureOffset(uniform_sampler1DArrayShadow, vec3(coord), int(0))

        // GLSL: textureOffset({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && float(0) == textureOffset(uniform_sampler1DArrayShadow, vec3(coord), int(0), float(0))

        // GLSL: textureOffset({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageSampleDrefImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && float(0) == textureOffset(uniform_sampler2DArrayShadow, vec4(coord), offset2D)

        // GLSL: texelFetch({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod %
        && gvec4(T.Element(0)) == texelFetch(gsampler1D, int(coord), int(0))

        // GLSL: texelFetch({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod %
        && gvec4(T.Element(0)) == texelFetch(gsampler2D, ivec2(coord), int(0))

        // GLSL: texelFetch({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod %
        && gvec4(T.Element(0)) == texelFetch(gsampler3D, ivec3(coord), int(0))

        // GLSL: texelFetch({{.*}}sampler2DRect
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]
        && gvec4(T.Element(0)) == texelFetch(gsampler2DRect, ivec2(coord))

        // GLSL: texelFetch({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod %
        && gvec4(T.Element(0)) == texelFetch(gsampler1DArray, ivec2(coord), int(0))

        // GLSL: texelFetch({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod %
        && gvec4(T.Element(0)) == texelFetch(gsampler2DArray, ivec3(coord), int(0))

        // GLSL: imageLoad({{.*}}samplerBuffer
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerBuffer
        // SPIR: OpImageRead {{.*}}[[LOAD]]
        && (gvec4(T.Element(0)) == texelFetch(gsamplerBuffer, int(coord)) || ignoreResultF32)

        // GLSL: texelFetch({{.*}}sampler2DMS
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DMS
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // S-PIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod %
        && gvec4(T.Element(0)) == texelFetch(gsampler2DMS, ivec2(coord), int(0))

        // GLSL: texelFetch({{.*}}sampler2DMSArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DMSArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // S-PIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod %
        && gvec4(T.Element(0)) == texelFetch(gsampler2DMSArray, ivec3(coord), int(0))

        // GLSL: texelFetchOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler1D, int(coord), int(0), int(0))

        // GLSL: texelFetchOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2D, ivec2(coord), int(0), offset2D)

        // GLSL: texelFetchOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler3D, ivec3(coord), int(0), offset3D)

        // GLSL: texelFetchOffset({{.*}}sampler2DRect
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2DRect, ivec2(coord), offset2D)

        // GLSL: texelFetchOffset({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler1DArray, ivec2(coord), int(0), int(0))

        // GLSL: texelFetchOffset({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: [[IMAGE:%[1-9][0-9]*]] = OpImage{{.*}}[[LOAD]]
        // SPIR: OpImageFetch {{.*}}[[IMAGE]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2DArray, ivec3(coord), int(0), offset2D)

        // GLSL: textureProjOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec2(coord), int(0))

        // GLSL: textureProjOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec2(coord), int(0), float(0))

        // GLSL: textureProjOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec4(coord), int(0))

        // GLSL: textureProjOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec4(coord), int(0), float(0))

        // GLSL: textureProjOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec3(coord), offset2D)

        // GLSL: textureProjOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec3(coord), offset2D, float(0))

        // GLSL: textureProjOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec4(coord), offset2D)

        // GLSL: textureProjOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec4(coord), offset2D, float(0))

        // GLSL: textureProjOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler3D, vec4(coord), offset3D)

        // GLSL: textureProjOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler3D, vec4(coord), offset3D, float(0))

        // GLSL-COUNT-2: textureProjOffset({{.*}}sampler2DRect
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleProjImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2DRect, vec3(coord), offset2D)
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2DRect, vec4(coord), offset2D)

        // GLSL: textureProjOffset({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && float(0) == textureProjOffset(uniform_sampler2DRectShadow, vec4(coord), offset2D)

        // GLSL: textureProjOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && float(0) == textureProjOffset(uniform_sampler1DShadow, vec4(coord), int(0))

        // GLSL: textureProjOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && float(0) == textureProjOffset(uniform_sampler1DShadow, vec4(coord), int(0), float(0))

        // GLSL: textureProjOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && (float(0) == textureProjOffset(uniform_sampler2DShadow, vec4(coord), offset2D) || ignoreResultF32)

        // GLSL: textureProjOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefImplicitLod {{.*}}[[LOAD]]{{.*}} Bias|ConstOffset %
        && (float(0) == textureProjOffset(uniform_sampler2DShadow, vec4(coord), offset2D, float(0)) || ignoreResultF32)

        // GLSL: textureLodOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureLodOffset(gsampler1D, float(coord), float(0), 0)

        // GLSL: textureLodOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureLodOffset(gsampler2D, vec2(coord), float(0), offset2D)

        // GLSL: textureLodOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureLodOffset(gsampler3D, vec3(coord), float(0), offset3D)

        // GLSL: textureLodOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && float(0) == textureLodOffset(uniform_sampler1DShadow, vec3(coord), float(0), 0)

        // GLSL: textureLodOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && (float(0) == textureLodOffset(uniform_sampler2DShadow, vec3(coord), float(0), offset2D) || ignoreResultF32)

        // GLSL: textureLodOffset({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureLodOffset(gsampler1DArray, vec2(coord), float(0), 0)

        // GLSL: textureLodOffset({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureLodOffset(gsampler2DArray, vec3(coord), float(0), offset2D)

        // GLSL: textureLodOffset({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && float(0) == textureLodOffset(uniform_sampler1DArrayShadow, vec3(coord), float(0), int(0))

        // GLSL-COUNT-2: textureProjLod({{.*}}sampler1D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureProjLod(gsampler1D, vec2(coord), float(0))
        && gvec4(T.Element(0)) == textureProjLod(gsampler1D, vec4(coord), float(0))

        // GLSL-COUNT-2: textureProjLod({{.*}}sampler2D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureProjLod(gsampler2D, vec3(coord), float(0))
        && gvec4(T.Element(0)) == textureProjLod(gsampler2D, vec4(coord), float(0))

        // GLSL: textureProjLod({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && gvec4(T.Element(0)) == textureProjLod(gsampler3D, vec4(coord), float(0))

        // GLSL: textureProjLod({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && float(0) == textureProjLod(uniform_sampler1DShadow, vec4(coord), float(0))

        // GLSL: textureProjLod({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod %
        && (float(0) == textureProjLod(uniform_sampler2DShadow, vec4(coord), float(0)) || ignoreResultF32)

        // GLSL-COUNT-2: textureProjLodOffset({{.*}}sampler1D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler1D, vec2(coord), float(0), 0)
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler1D, vec4(coord), float(0), 0)

        // GLSL-COUNT-2: textureProjLodOffset({{.*}}sampler2D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler2D, vec3(coord), float(0), offset2D)
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler2D, vec4(coord), float(0), offset2D)

        // GLSL: textureProjLodOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler3D, vec4(coord), float(0), offset3D)

        // GLSL: textureProjLodOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && float(0) == textureProjLodOffset(uniform_sampler1DShadow, vec4(coord), float(0), int(0))

        // GLSL: textureProjLodOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Lod|ConstOffset %
        && (float(0) == textureProjLodOffset(uniform_sampler2DShadow, vec4(coord), float(0), offset2D) || ignoreResultF32)

        // GLSL: textureGrad({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsampler1D, float(coord), float(0), float(0))

        // GLSL: textureGrad({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsampler2D, vec2(coord), vec2(0), vec2(0))

        // GLSL: textureGrad({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsampler3D, vec3(coord), vec3(0), vec3(0))

        // GLSL: textureGrad({{.*}}samplerCube
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsamplerCube, vec3(coord), vec3(0), vec3(0))

        // GLSL: textureGrad({{.*}}sampler2DRect
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsampler2DRect, vec2(coord), vec2(0), vec2(0))

        // GLSL: textureGrad({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && float(0) == textureGrad(uniform_sampler2DRectShadow, vec3(coord), vec2(0), vec2(0))

        // GLSL: textureGrad({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && float(0) == textureGrad(uniform_sampler1DShadow, vec3(coord), float(0), float(0))

        // GLSL: textureGrad({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsampler1DArray, vec2(coord), float(0), float(0))

        // GLSL: textureGrad({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsampler2DArray, vec3(coord), vec2(0), vec2(0))

        // GLSL: textureGrad({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && float(0) == textureGrad(uniform_sampler1DArrayShadow, vec3(coord), float(0), float(0))

        // GLSL: textureGrad({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && (float(0) == textureGrad(uniform_sampler2DShadow, vec3(coord), vec2(0), vec2(0)) || ignoreResultF32)

        // GLSL: textureGrad({{.*}}samplerCubeShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && (float(0) == textureGrad(uniform_samplerCubeShadow, vec4(coord), vec3(0), vec3(0)) || ignoreResultF32)

        // GLSL: textureGrad({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && float(0) == textureGrad(uniform_sampler2DArrayShadow, vec4(coord), vec2(0), vec2(0))

        // GLSL: textureGrad({{.*}}samplerCubeArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureGrad(gsamplerCubeArray, vec4(coord), vec3(0), vec3(0))

        // GLSL: textureGradOffset({{.*}}sampler1D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureGradOffset(gsampler1D, float(coord), float(0), float(0), 0)

        // GLSL: textureGradOffset({{.*}}sampler2D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureGradOffset(gsampler2D, vec2(coord), vec2(0), vec2(0), offset2D)

        // GLSL: textureGradOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureGradOffset(gsampler3D, vec3(coord), vec3(0), vec3(0), offset3D)

        // GLSL: textureGradOffset({{.*}}sampler2DRect
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureGradOffset(gsampler2DRect, vec2(coord), vec2(0), vec2(0), offset2D)

        // GLSL: textureGradOffset({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && float(0) == textureGradOffset(uniform_sampler2DRectShadow, vec3(coord), vec2(0), vec2(0), offset2D)

        // GLSL: textureGradOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && float(0) == textureGradOffset(uniform_sampler1DShadow, vec3(coord), float(0), float(0), int(0))

        // GLSL: textureGradOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && (float(0) == textureGradOffset(uniform_sampler2DShadow, vec3(coord), vec2(0), vec2(0), offset2D) || ignoreResultF32)

        // GLSL: textureGradOffset({{.*}}sampler2DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureGradOffset(gsampler2DArray, vec3(coord), vec2(0), vec2(0), offset2D)

        // GLSL: textureGradOffset({{.*}}sampler1DArray
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArray
        // SPIR: OpImageSampleExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureGradOffset(gsampler1DArray, vec2(coord), float(0), float(0), int(0))

        // GLSL: textureGradOffset({{.*}}sampler1DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DArrayShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && float(0) == textureGradOffset(uniform_sampler1DArrayShadow, vec3(coord), float(0), float(0), int(0))

        // GLSL: textureGradOffset({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageSampleDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && float(0) == textureGradOffset(uniform_sampler2DArrayShadow, vec4(coord), vec2(0), vec2(0), offset2D)

        // GLSL-COUNT-2: textureProjGrad({{.*}}sampler1D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureProjGrad(gsampler1D, vec2(coord), float(0), float(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler1D, vec4(coord), float(0), float(0))

        // GLSL-COUNT-2: textureProjGrad({{.*}}sampler2D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2D, vec3(coord), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2D, vec4(coord), vec2(0), vec2(0))

        // GLSL: textureProjGrad({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureProjGrad(gsampler3D, vec4(coord), vec3(0), vec3(0))

        // GLSL-COUNT-2: textureProjGrad({{.*}}sampler2DRect
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2DRect, vec3(coord), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2DRect, vec4(coord), vec2(0), vec2(0))

        // GLSL: textureProjGrad({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && float(0) == textureProjGrad(uniform_sampler2DRectShadow, vec4(coord), vec2(0), vec2(0))

        // GLSL: textureProjGrad({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && float(0) == textureProjGrad(uniform_sampler1DShadow, vec4(coord), float(0), float(0))

        // GLSL: textureProjGrad({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad %
        && (float(0) == textureProjGrad(uniform_sampler2DShadow, vec4(coord), vec2(0), vec2(0)) || ignoreResultF32)

        // GLSL-COUNT-2: textureProjGradOffset({{.*}}sampler1D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler1D, vec2(coord), float(0), float(0), 0)
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler1D, vec4(coord), float(0), float(0), 0)

        // GLSL-COUNT-2: textureProjGradOffset({{.*}}sampler2D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2D, vec3(coord), vec2(0), vec2(0), offset2D)
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2D, vec4(coord), vec2(0), vec2(0), offset2D)

        // GLSL: textureProjGradOffset({{.*}}sampler3D
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler3D
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler3D, vec4(coord), vec3(0), vec3(0), offset3D)

        // GLSL-COUNT-2: textureProjGradOffset({{.*}}sampler2DRect
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageSampleProjExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2DRect, vec3(coord), vec2(0), vec2(0), offset2D)
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2DRect, vec4(coord), vec2(0), vec2(0), offset2D)

        // GLSL: textureProjGradOffset({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && float(0) == textureProjGradOffset(uniform_sampler2DRectShadow, vec4(coord), vec2(0), vec2(0), offset2D)

        // GLSL: textureProjGradOffset({{.*}}sampler1DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler1DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && float(0) == textureProjGradOffset(uniform_sampler1DShadow, vec4(coord), float(0), float(0), int(0))

        // GLSL: textureProjGradOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageSampleProjDrefExplicitLod {{.*}}[[LOAD]]{{.*}} Grad|ConstOffset %
        && (float(0) == textureProjGradOffset(uniform_sampler2DShadow, vec4(coord), vec2(0), vec2(0), offset2D) || ignoreResultF32)
        
        // 8.9.4. Texture Gather Functions

        // GLSL-COUNT-2: textureGather({{.*}}sampler2D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageGather {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureGather(gsampler2D, vec2(coord))
        && gvec4(T.Element(0)) == textureGather(gsampler2D, vec2(coord), int(0))

        // GLSL-COUNT-2: textureGather({{.*}}sampler2DArray
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageGather {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureGather(gsampler2DArray, vec3(coord))
        && gvec4(T.Element(0)) == textureGather(gsampler2DArray, vec3(coord), int(0))

        // GLSL-COUNT-2: textureGather({{.*}}samplerCube
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCube
        // SPIR: OpImageGather {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureGather(gsamplerCube, vec3(coord))
        && gvec4(T.Element(0)) == textureGather(gsamplerCube, vec3(coord), int(0))

        // GLSL-COUNT-2: textureGather({{.*}}samplerCubeArray
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArray
        // SPIR: OpImageGather {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureGather(gsamplerCubeArray, vec4(coord))
        && gvec4(T.Element(0)) == textureGather(gsamplerCubeArray, vec4(coord), int(0))

        // GLSL-COUNT-2: textureGather({{.*}}sampler2DRect
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageGather {{.*}}[[LOAD]]
        && gvec4(T.Element(0)) == textureGather(gsampler2DRect, vec2(coord))
        && gvec4(T.Element(0)) == textureGather(gsampler2DRect, vec2(coord), int(0))

        // GLSL: textureGather({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]
        && (vec4(0) == textureGather(uniform_sampler2DShadow, vec2(coord), float(0)) || ignoreResultF32)

        // GLSL: textureGather({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]
        && vec4(0) == textureGather(uniform_sampler2DArrayShadow, vec3(coord), float(0))

        // GLSL: textureGather({{.*}}samplerCubeShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]
        && (vec4(0) == textureGather(uniform_samplerCubeShadow, vec3(coord), float(0)) || ignoreResultF32)

        // GLSL: textureGather({{.*}}samplerCubeArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}samplerCubeArrayShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]
        && vec4(0) == textureGather(uniform_samplerCubeArrayShadow, vec4(coord), float(0))

        // GLSL: textureGather({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]
        && vec4(0) == textureGather(uniform_sampler2DRectShadow, vec2(coord), float(0))

        // GLSL-COUNT-2: textureGatherOffset({{.*}}sampler2D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageGather {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2D, vec2(coord), offset2D)
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2D, vec2(coord), offset2D, int(0))

        // GLSL-COUNT-2: textureGatherOffset({{.*}}sampler2DArray
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageGather {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DArray, vec3(coord), offset2D)
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DArray, vec3(coord), offset2D, int(0))

        // GLSL: textureGatherOffset({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && (vec4(0) == textureGatherOffset(uniform_sampler2DShadow, vec2(coord), float(0), offset2D) || ignoreResultF32)

        // GLSL: textureGatherOffset({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && vec4(0) == textureGatherOffset(uniform_sampler2DArrayShadow, vec3(coord), float(0), offset2D)

        // GLSL-COUNT-2: textureGatherOffset({{.*}}sampler2DRect
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageGather {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DRect, vec2(coord), offset2D)
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DRect, vec2(coord), offset2D, int(0))

        // GLSL: textureGatherOffset({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]{{.*}} ConstOffset %
        && vec4(0) == textureGatherOffset(uniform_sampler2DRectShadow, vec2(coord), float(0), offset2D)

        // GLSL-COUNT-2: textureGatherOffsets({{.*}}sampler2D
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2D
        // SPIR: OpImageGather {{.*}}[[LOAD]]{{.*}} ConstOffsets %
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2D, vec2(coord), offsets)
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2D, vec2(coord), offsets, int(0))

        // GLSL-COUNT-2: textureGatherOffsets({{.*}}sampler2DArray
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArray
        // SPIR: OpImageGather {{.*}}[[LOAD]]{{.*}} ConstOffsets %
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DArray, vec3(coord), offsets)
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DArray, vec3(coord), offsets, int(0))

        // GLSL: textureGatherOffsets({{.*}}sampler2DShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]{{.*}} ConstOffsets %
        && (vec4(0) == textureGatherOffsets(uniform_sampler2DShadow, vec2(coord), float(0), offsets) || ignoreResultF32)

        // GLSL: textureGatherOffsets({{.*}}sampler2DArrayShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DArrayShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]{{.*}} ConstOffsets %
        && vec4(0) == textureGatherOffsets(uniform_sampler2DArrayShadow, vec3(coord), float(0), offsets)

        // GLSL-COUNT-2: textureGatherOffsets({{.*}}sampler2DRect
        // SPIR-COUNT-2: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRect
        // SPIR: OpImageGather {{.*}}[[LOAD]]{{.*}} ConstOffsets %
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DRect, vec2(coord), offsets)
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DRect, vec2(coord), offsets, int(0))

        // GLSL: textureGatherOffsets({{.*}}sampler2DRectShadow
        // SPIR: [[LOAD:%[1-9][0-9]*]] = OpLoad{{.*}}sampler2DRectShadow
        // SPIR: OpImageDrefGather {{.*}}[[LOAD]]{{.*}} ConstOffsets %
        && vec4(0) == textureGatherOffsets(uniform_sampler2DRectShadow, vec2(coord), float(0), offsets)

        // 8.9.5. Compatibility Profile Texture Functions
        // Compatibility functions work for GLSL but not for SPIR-V.
        // We cannot test the SPIR-V asm here for these functions.
        // We also cannot test GLSL source for them because we don't directly
        // call them. If we do, the spirv-via-glsl path cannot work.

        && vec4(0) == texture1D(uniform_sampler1D, float(coord))
        && vec4(0) == texture1D(uniform_sampler1D, float(coord), float(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec2(coord))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec2(coord), float(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec4(coord))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec4(coord), float(0))
        && vec4(0) == texture1DLod(uniform_sampler1D, float(coord), float(0))
        && vec4(0) == texture1DProjLod(uniform_sampler1D, vec2(coord), float(0))
        && vec4(0) == texture1DProjLod(uniform_sampler1D, vec4(coord), float(0))
        && vec4(0) == texture2D(uniform_sampler2D, vec2(coord))
        && vec4(0) == texture2D(uniform_sampler2D, vec2(coord), float(0))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec3(coord))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec3(coord), float(0))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec4(coord))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec4(coord), float(0))
        && vec4(0) == texture2DLod(uniform_sampler2D, vec2(coord), float(0))
        && vec4(0) == texture2DProjLod(uniform_sampler2D, vec3(coord), float(0))
        && vec4(0) == texture2DProjLod(uniform_sampler2D, vec4(coord), float(0))
        && vec4(0) == texture3D(uniform_sampler3D, vec3(coord))
        && vec4(0) == texture3D(uniform_sampler3D, vec3(coord), float(0))
        && vec4(0) == texture3DProj(uniform_sampler3D, vec4(coord))
        && vec4(0) == texture3DProj(uniform_sampler3D, vec4(coord), float(0))
        && vec4(0) == texture3DLod(uniform_sampler3D, vec3(coord), float(0))
        && vec4(0) == texture3DProjLod(uniform_sampler3D, vec4(coord), float(0))
        && vec4(0) == textureCube(uniform_samplerCube, vec3(coord))
        && vec4(0) == textureCube(uniform_samplerCube, vec3(coord), float(0))
        && vec4(0) == textureCubeLod(uniform_samplerCube, vec3(coord), float(0))
        && vec4(0) == shadow1D(uniform_sampler1DShadow, vec3(coord))
        && vec4(0) == shadow1D(uniform_sampler1DShadow, vec3(coord), float(0))
        && (vec4(0) == shadow2D(uniform_sampler2DShadow, vec3(coord)) || ignoreResultF32)
        && (vec4(0) == shadow2D(uniform_sampler2DShadow, vec3(coord), float(0)) || ignoreResultF32)
        && vec4(0) == shadow1DProj(uniform_sampler1DShadow, vec4(coord))
        && vec4(0) == shadow1DProj(uniform_sampler1DShadow, vec4(coord), float(0))
        && (vec4(0) == shadow2DProj(uniform_sampler2DShadow, vec4(coord)) || ignoreResultF32)
        && (vec4(0) == shadow2DProj(uniform_sampler2DShadow, vec4(coord), float(0)) || ignoreResultF32)
        && vec4(0) == shadow1DLod(uniform_sampler1DShadow, vec3(coord), float(0))
        && (vec4(0) == shadow2DLod(uniform_sampler2DShadow, vec3(coord), float(0)) || ignoreResultF32)
        && vec4(0) == shadow1DProjLod(uniform_sampler1DShadow, vec4(coord), float(0))
        && (vec4(0) == shadow2DProjLod(uniform_sampler2DShadow, vec4(coord), float(0)) || ignoreResultF32)
        ;
}

#if defined(HLSL)
[require(hlsl, sm_5_1, compute_fragment)]
#elif defined(GLSL)
[require(glsl, sm_5_1, compute_fragment)]
#elif defined(SPIRV)
[require(spirv, sm_5_1, compute_fragment)]
#elif defined(CUDA)
[require(cuda, sm_5_1, compute_fragment)]
#endif
__extern_cpp void TEST_main()
{
    // GLSL: void main(
    // GLSL_SPIRV: OpEntryPoint
    // HLSL: void TEST_main(
    // CUDA: void TEST_main(
    // CPP: void _TEST_main(
    // BUF: 1

    // Some of the functional testing result is incorrect due to
    // the problem on our testing framework. We will ignore them
    // until we address the problems.
    // The problem is mostly with 2DShadow, CubeShadow and buffer.
    bool ignoreResult = (outputBuffer.result == 0); // always true

    bool result = true
        && textureFuncs(
            uniform_sampler1D
            , uniform_sampler2D
            , uniform_sampler2DRect
            , uniform_sampler3D
            , uniform_samplerCube
            , uniform_sampler1DArray
            , uniform_sampler2DArray
            , uniform_samplerCubeArray
            , uniform_samplerBuffer
            , uniform_sampler2DMS
            , uniform_sampler2DMSArray
            , ignoreResult
            )
#if !defined(HLSL)
        && textureFuncs(
            uniform_isampler1D
            , uniform_isampler2D
            , uniform_isampler2DRect
            , uniform_isampler3D
            , uniform_isamplerCube
            , uniform_isampler1DArray
            , uniform_isampler2DArray
            , uniform_isamplerCubeArray
            , uniform_isamplerBuffer
            , uniform_isampler2DMS
            , uniform_isampler2DMSArray
            , ignoreResult
            )
        && textureFuncs(
            uniform_usampler1D
            , uniform_usampler2D
            , uniform_usampler2DRect
            , uniform_usampler3D
            , uniform_usamplerCube
            , uniform_usampler1DArray
            , uniform_usampler2DArray
            , uniform_usamplerCubeArray
            , uniform_usamplerBuffer
            , uniform_usampler2DMS
            , uniform_usampler2DMSArray
            , ignoreResult
            )
#endif
        ;

    outputBuffer.result = int(result);
}

[shader("compute")]
[DerivativeGroupLinear]
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TEST_main();
}

[shader("fragment")]
void fragMain()
{
    TEST_main();
}
