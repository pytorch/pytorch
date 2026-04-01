#version 450 core
#extension GL_EXT_texture_shadow_lod : enable

//TEST:SIMPLE(filecheck=CHECK_GLSL): -allow-glsl -stage compute -entry computeMain -target glsl -DCOMPUTE -ignore-capabilities
//TEST:SIMPLE(filecheck=CHECK_GLSL): -allow-glsl -stage fragment -entry computeMain -target glsl -ignore-capabilities
//RWStructuredBuffer<float> outputBuffer;
//layout(location = 0) out vec4 o_color;

buffer MyBlockName
{
    int result;
} outputBuffer;

uniform sampler1DShadow uniform_sampler1DShadow;
uniform sampler2DShadow uniform_sampler2DShadow;
uniform samplerCubeShadow uniform_samplerCubeShadow;

uniform sampler1DArrayShadow uniform_sampler1DArrayShadow;
uniform sampler2DArrayShadow uniform_sampler2DArrayShadow;
uniform sampler2DRectShadow uniform_sampler2DRectShadow;
uniform samplerCubeArrayShadow uniform_samplerCubeArrayShadow;

uniform sampler1D uniform_sampler1D;
uniform sampler2D uniform_sampler2D;
uniform sampler2DRect uniform_sampler2DRect;
uniform sampler3D uniform_sampler3D;
uniform samplerCube uniform_samplerCube;
uniform sampler1DArray uniform_sampler1DArray;
uniform sampler2DArray uniform_sampler2DArray;
uniform samplerCubeArray uniform_samplerCubeArray;
uniform samplerBuffer uniform_samplerBuffer;

uniform isampler1D uniform_isampler1D;
uniform isampler2D uniform_isampler2D;
uniform isampler2DRect uniform_isampler2DRect;
uniform isampler3D uniform_isampler3D;
uniform isamplerCube uniform_isamplerCube;
uniform isampler1DArray uniform_isampler1DArray;
uniform isampler2DArray uniform_isampler2DArray;
uniform isamplerCubeArray uniform_isamplerCubeArray;
uniform isamplerBuffer uniform_isamplerBuffer;

uniform usampler1D uniform_usampler1D;
uniform usampler2D uniform_usampler2D;
uniform usampler2DRect uniform_usampler2DRect;
uniform usampler3D uniform_usampler3D;
uniform usamplerCube uniform_usamplerCube;
uniform usampler1DArray uniform_usampler1DArray;
uniform usampler2DArray uniform_usampler2DArray;
uniform usamplerCubeArray uniform_usamplerCubeArray;
uniform usamplerBuffer uniform_usamplerBuffer;

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
)
{
    typealias gvec4 = vector<T.Element,4>;

    constexpr ivec2 ivec2_0 = ivec2(0);

    return true
        // 8.9.1. Texture Query Functions
        && int(0) == textureSize(gsampler1D, int(0))
        && ivec2(0) == textureSize(gsampler2D, int(0))
        && ivec3(0) == textureSize(gsampler3D, int(0))
        && ivec2(0) == textureSize(gsamplerCube, int(0))
        && int(0) == textureSize(uniform_sampler1DShadow, int(0))
        && ivec2(0) == textureSize(uniform_sampler2DShadow, int(0))
        && ivec2(0) == textureSize(uniform_samplerCubeShadow, int(0))
        && ivec3(0) == textureSize(gsamplerCubeArray, int(0))
        && ivec3(0) == textureSize(uniform_samplerCubeArrayShadow, int(0))
        && ivec2(0) == textureSize(gsampler2DRect)
        && ivec2(0) == textureSize(uniform_sampler2DRectShadow)
        && ivec2(0) == textureSize(gsampler1DArray, int(0))
        && ivec2(0) == textureSize(uniform_sampler1DArrayShadow, int(0))
        && ivec3(0) == textureSize(gsampler2DArray, int(0))
        && ivec3(0) == textureSize(uniform_sampler2DArrayShadow, int(0))
        && int(0) == textureSize(gsamplerBuffer)
        && vec2(0) == textureQueryLod(gsampler1D, float(0))
        && vec2(0) == textureQueryLod(gsampler2D, vec2(0))
        && vec2(0) == textureQueryLod(gsampler3D, vec3(0))
        && vec2(0) == textureQueryLod(gsamplerCube, vec3(0))
        && vec2(0) == textureQueryLod(gsampler1DArray, float(0))
        && vec2(0) == textureQueryLod(gsampler2DArray, vec2(0))
        && vec2(0) == textureQueryLod(gsamplerCubeArray, vec3(0))
        && vec2(0) == textureQueryLod(uniform_sampler1DShadow, float(0))
        && vec2(0) == textureQueryLod(uniform_sampler2DShadow, vec2(0))
        && vec2(0) == textureQueryLod(uniform_samplerCubeShadow, vec3(0))
        && vec2(0) == textureQueryLod(uniform_sampler1DArrayShadow, float(0))
        && vec2(0) == textureQueryLod(uniform_sampler2DArrayShadow, vec2(0))
        && vec2(0) == textureQueryLod(uniform_samplerCubeArrayShadow, vec3(0))
        && int(0) == textureQueryLevels(gsampler1D)
        && int(0) == textureQueryLevels(gsampler2D)
        && int(0) == textureQueryLevels(gsampler3D)
        && int(0) == textureQueryLevels(gsamplerCube)
        && int(0) == textureQueryLevels(gsampler1DArray)
        && int(0) == textureQueryLevels(gsampler2DArray)
        && int(0) == textureQueryLevels(gsamplerCubeArray)
        && int(0) == textureQueryLevels(uniform_sampler1DShadow)
        && int(0) == textureQueryLevels(uniform_sampler2DShadow)
        && int(0) == textureQueryLevels(uniform_samplerCubeShadow)
        && int(0) == textureQueryLevels(uniform_sampler1DArrayShadow)
        && int(0) == textureQueryLevels(uniform_sampler2DArrayShadow)
        && int(0) == textureQueryLevels(uniform_samplerCubeArrayShadow)
        // 8.9.2. Texel Lookup Functions
        && gvec4(T.Element(0)) == texture(gsampler1D, float(0))
        && gvec4(T.Element(0)) == texture(gsampler1D, float(0), float(0))
        && gvec4(T.Element(0)) == texture(gsampler2D, vec2(0))
        && gvec4(T.Element(0)) == texture(gsampler2D, vec2(0), float(0))
        && gvec4(T.Element(0)) == texture(gsampler3D, vec3(0))
        && gvec4(T.Element(0)) == texture(gsampler3D, vec3(0), float(0))
        && gvec4(T.Element(0)) == texture(gsamplerCube, vec3(0) )
        && gvec4(T.Element(0)) == texture(gsamplerCube, vec3(0), float(0))
        && float(0) == texture(uniform_sampler1DShadow, vec3(0))
        && float(0) == texture(uniform_sampler1DShadow, vec3(0), float(0))
        && float(0) == texture(uniform_sampler2DShadow, vec3(0))
        && float(0) == texture(uniform_sampler2DShadow, vec3(0), float(0))
        && float(0) == texture(uniform_samplerCubeShadow, vec4(0))
        && float(0) == texture(uniform_samplerCubeShadow, vec4(0), float(0))
        && gvec4(T.Element(0)) == texture(gsampler2DArray, vec3(0))
        && gvec4(T.Element(0)) == texture(gsampler2DArray, vec3(0), float(0))
        && gvec4(T.Element(0)) == texture(gsamplerCubeArray, vec4(0) )
        && gvec4(T.Element(0)) == texture(gsamplerCubeArray, vec4(0), float(0))
        && gvec4(T.Element(0)) == texture(gsampler1DArray, vec2(0))
        && gvec4(T.Element(0)) == texture(gsampler1DArray, vec2(0), float(0))
        && float(0) == texture(uniform_sampler1DArrayShadow, vec3(0))
        && float(0) == texture(uniform_sampler1DArrayShadow, vec3(0), float(0))
        && float(0) == texture(uniform_sampler2DArrayShadow, vec4(0))
        && gvec4(T.Element(0)) == texture(gsampler2DRect, vec2(0))
        && float(0) == texture(uniform_sampler2DRectShadow, vec3(0))
        && float(0) == texture(uniform_samplerCubeArrayShadow, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec2(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec2(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec4(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec3(0))
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec3(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec4(0))
        && gvec4(T.Element(0)) == textureProj(gsampler2D, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler3D, vec4(0))
        && gvec4(T.Element(0)) == textureProj(gsampler3D, vec4(0), float(0))
        && float(0) == textureProj(uniform_sampler1DShadow, vec4(0))
        && float(0) == textureProj(uniform_sampler1DShadow, vec4(0), float(0))
        && float(0) == textureProj(uniform_sampler2DShadow, vec4(0))
        && float(0) == textureProj(uniform_sampler2DShadow, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler2DRect, vec3(0))
        && gvec4(T.Element(0)) == textureProj(gsampler2DRect, vec4(0))
        && float(0) == textureProj(uniform_sampler2DRectShadow, vec4(0))
        && gvec4(T.Element(0)) == textureLod(gsampler1D, float(0), float(0))
        && gvec4(T.Element(0)) == textureLod(gsampler2D, vec2(0), float(0))
        && gvec4(T.Element(0)) == textureLod(gsampler3D, vec3(0), float(0))
        && gvec4(T.Element(0)) == textureLod(gsamplerCube, vec3(0), float(0))
        && float(0) == textureLod(uniform_sampler2DShadow, vec3(0), float(0))
        && float(0) == textureLod(uniform_sampler1DShadow, vec3(0), float(0))
        && gvec4(T.Element(0)) == textureLod(gsampler1DArray, vec2(0), float(0))
        && float(0) == textureLod(uniform_sampler1DArrayShadow, vec3(0), float(0))
        && gvec4(T.Element(0)) == textureLod(gsampler2DArray, vec3(0), float(0))
        && gvec4(T.Element(0)) == textureLod(gsamplerCubeArray, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureOffset(gsampler1D, float(0), __LINE__)
        && gvec4(T.Element(0)) == textureOffset(gsampler1D, float(0), __LINE__, float(0))
        && gvec4(T.Element(0)) == textureOffset(gsampler2D, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureOffset(gsampler2D, vec2(0), { __LINE__ }, float(0))
        && gvec4(T.Element(0)) == textureOffset(gsampler3D, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureOffset(gsampler3D, vec3(0), { __LINE__ }, float(0))
        && float(0) == textureOffset(uniform_sampler2DShadow, vec3(0), { __LINE__ })
        && float(0) == textureOffset(uniform_sampler2DShadow, vec3(0), { __LINE__ }, float(0))
        && gvec4(T.Element(0)) == textureOffset(gsampler2DRect, vec2(0), { __LINE__ })
        && float(0) == textureOffset(uniform_sampler2DRectShadow, vec3(0), { __LINE__ })
        && float(0) == textureOffset(uniform_sampler1DShadow, vec3(0), __LINE__)
        && float(0) == textureOffset(uniform_sampler1DShadow, vec3(0), __LINE__, float(0))
        && gvec4(T.Element(0)) == textureOffset(gsampler1DArray, vec2(0), __LINE__)
        && gvec4(T.Element(0)) == textureOffset(gsampler1DArray, vec2(0), __LINE__, float(0))
        && gvec4(T.Element(0)) == textureOffset(gsampler2DArray, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureOffset(gsampler2DArray, vec3(0), { __LINE__ }, float(0))
        && float(0) == textureOffset(uniform_sampler1DArrayShadow, vec3(0), __LINE__)
        && float(0) == textureOffset(uniform_sampler1DArrayShadow, vec3(0), __LINE__, float(0))
        && float(0) == textureOffset(uniform_sampler2DArrayShadow, vec4(0), { __LINE__ })
        && gvec4(T.Element(0)) == texelFetch(gsampler1D, int(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler2D, ivec2(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler3D, ivec3(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler2DRect, ivec2(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler1DArray, ivec2(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler2DArray, ivec3(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsamplerBuffer, int(0))
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler1D, int(0), int(0), __LINE__)
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2D, ivec2(0), int(0), { __LINE__ })
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler3D, ivec3(0), int(0), { __LINE__ })
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2DRect, ivec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler1DArray, ivec2(0), int(0), __LINE__)
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2DArray, ivec3(0), int(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec2(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec2(0), __LINE__, float(0))
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec4(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec4(0), __LINE__,float(0))
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec3(0), { __LINE__ }, float(0))
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec4(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2D, vec4(0), { __LINE__ }, float(0))
        && gvec4(T.Element(0)) == textureProjOffset(gsampler3D, vec4(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjOffset(gsampler3D, vec4(0), { __LINE__ }, float(0))
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2DRect, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjOffset(gsampler2DRect, vec4(0), { __LINE__ })
        && float(0) == textureProjOffset(uniform_sampler2DRectShadow, vec4(0), { __LINE__ })
        && float(0) == textureProjOffset(uniform_sampler1DShadow, vec4(0), __LINE__)
        && float(0) == textureProjOffset(uniform_sampler1DShadow, vec4(0), __LINE__, float(0))
        && float(0) == textureProjOffset(uniform_sampler2DShadow, vec4(0), { __LINE__ })
        && float(0) == textureProjOffset(uniform_sampler2DShadow, vec4(0), { __LINE__ }, float(0))
        && gvec4(T.Element(0)) == textureLodOffset(gsampler1D, float(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureLodOffset(gsampler2D, vec2(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureLodOffset(gsampler3D, vec3(0), float(0), { __LINE__ })
        && float(0) == textureLodOffset(uniform_sampler1DShadow, vec3(0), float(0), __LINE__)
        && float(1) == textureLodOffset(uniform_sampler2DShadow, vec3(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureLodOffset(gsampler1DArray, vec2(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureLodOffset(gsampler2DArray, vec3(0), float(0), { __LINE__ })
        && float(0) == textureLodOffset(uniform_sampler1DArrayShadow, vec3(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjLod(gsampler1D, vec2(0), float(0))
        && gvec4(T.Element(0)) == textureProjLod(gsampler1D, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProjLod(gsampler2D, vec3(0), float(0))
        && gvec4(T.Element(0)) == textureProjLod(gsampler2D, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProjLod(gsampler3D, vec4(0), float(0))
        && float(0) == textureProjLod(uniform_sampler1DShadow, vec4(0), float(0))
        && float(0) == textureProjLod(uniform_sampler2DShadow, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler1D, vec2(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler1D, vec4(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler2D, vec3(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler2D, vec4(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler3D, vec4(0), float(0), { __LINE__ })
        && float(0) == textureProjLodOffset(uniform_sampler1DShadow, vec4(0), float(0), __LINE__)
        && float(0) == textureProjLodOffset(uniform_sampler2DShadow, vec4(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGrad(gsampler1D, float(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureGrad(gsampler2D, vec2(0), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureGrad(gsampler3D, vec3(0), vec3(0), vec3(0))
        && gvec4(T.Element(0)) == textureGrad(gsamplerCube, vec3(0), vec3(0), vec3(0))
        && gvec4(T.Element(0)) == textureGrad(gsampler2DRect, vec2(0), vec2(0), vec2(0))
        && float(0) == textureGrad(uniform_sampler2DRectShadow, vec3(0), vec2(0), vec2(0))
        && float(0) == textureGrad(uniform_sampler1DShadow, vec3(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureGrad(gsampler1DArray, vec2(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureGrad(gsampler2DArray, vec3(0), vec2(0), vec2(0))
        && float(0) == textureGrad(uniform_sampler1DArrayShadow, vec3(0), float(0), float(0))
        && float(0) == textureGrad(uniform_sampler2DShadow, vec3(0), vec2(0), vec2(0))
        && float(0) == textureGrad(uniform_samplerCubeShadow, vec4(0), vec3(0), vec3(0))
        && float(0) == textureGrad(uniform_sampler2DArrayShadow, vec4(0), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureGrad(gsamplerCubeArray, vec4(0), vec3(0), vec3(0))
        && gvec4(T.Element(0)) == textureGradOffset(gsampler1D, float(0), float(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureGradOffset(gsampler2D, vec2(0), vec2(0), vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGradOffset(gsampler3D, vec3(0), vec3(0), vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGradOffset(gsampler2DRect, vec2(0), vec2(0), vec2(0), { __LINE__ })
        && float(0) == textureGradOffset(uniform_sampler2DRectShadow, vec3(0), vec2(0), vec2(0), { __LINE__ })
        && float(0) == textureGradOffset(uniform_sampler1DShadow, vec3(0), float(0), float(0), __LINE__)
        && float(0) == textureGradOffset(uniform_sampler2DShadow, vec3(0), vec2(0), vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGradOffset(gsampler2DArray, vec3(0), vec2(0), vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGradOffset(gsampler1DArray, vec2(0), float(0), float(0), __LINE__)
        && float(0) == textureGradOffset(uniform_sampler1DArrayShadow, vec3(0), float(0), float(0), __LINE__)
        && float(0) == textureGradOffset(uniform_sampler2DArrayShadow, vec4(0), vec2(0), vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjGrad(gsampler1D, vec2(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler1D, vec4(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2D, vec3(0), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2D, vec4(0), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler3D, vec4(0), vec3(0), vec3(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2DRect, vec3(0), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler2DRect, vec4(0), vec2(0), vec2(0))
        && float(0) == textureProjGrad(uniform_sampler2DRectShadow, vec4(0), vec2(0), vec2(0))
        && float(0) == textureProjGrad(uniform_sampler1DShadow, vec4(0), float(0), float(0))
        && float(0) == textureProjGrad(uniform_sampler2DShadow, vec4(0), vec2(0), vec2(0))
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler1D, vec2(0), float(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler1D, vec4(0), float(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2D, vec3(0), vec2(0), vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2D, vec4(0), vec2(0), vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler3D, vec4(0), vec3(0), vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2DRect, vec3(0), vec2(0), vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler2DRect, vec4(0), vec2(0), vec2(0), { __LINE__ })
        && float(0) == textureProjGradOffset(uniform_sampler2DRectShadow, vec4(0), vec2(0), vec2(0), { __LINE__ })
        && float(0) == textureProjGradOffset(uniform_sampler1DShadow, vec4(0), float(0), float(0), __LINE__)
        && float(0) == textureProjGradOffset(uniform_sampler2DShadow, vec4(0), vec2(0), vec2(0), { __LINE__ })
        
        // 8.9.4. Texture Gather Functions
        && gvec4(T.Element(0)) == textureGather(gsampler2D, vec2(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2D, vec2(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DArray, vec3(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DArray, vec3(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCube, vec3(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCube, vec3(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCubeArray, vec4(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCubeArray, vec4(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DRect, vec2(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DRect, vec2(0), int(0))
        && vec4(0) == textureGather(uniform_sampler2DShadow, vec2(0), float(0))
        && vec4(0) == textureGather(uniform_sampler2DArrayShadow, vec3(0), float(0))
        && vec4(0) == textureGather(uniform_samplerCubeShadow, vec3(0), float(0))
        && vec4(0) == textureGather(uniform_samplerCubeArrayShadow, vec4(0), float(0))
        && vec4(0) == textureGather(uniform_sampler2DRectShadow, vec2(0), float(0))
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2D, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2D, vec2(0), { __LINE__ }, int(0))
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DArray, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DArray, vec3(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffset(uniform_sampler2DShadow, vec2(0), float(0), { __LINE__ })
        && vec4(0) == textureGatherOffset(uniform_sampler2DArrayShadow, vec3(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DRect, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DRect, vec2(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffset(uniform_sampler2DRectShadow, vec2(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2D, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2D, vec2(0), { __LINE__ }, int(0))
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DArray, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DArray, vec3(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffsets(uniform_sampler2DShadow, vec2(0), float(0), { __LINE__ })
        && vec4(0) == textureGatherOffsets(uniform_sampler2DArrayShadow, vec3(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DRect, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DRect, vec2(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffsets(uniform_sampler2DRectShadow, vec2(0), float(0), { __LINE__ })

        // 8.9.5. Compatibility Profile Texture Functions
        && vec4(0) == texture1D(uniform_sampler1D, float(0))
        && vec4(0) == texture1D(uniform_sampler1D, float(0), float(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec2(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec2(0), float(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec4(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec4(0), float(0))
        && vec4(0) == texture1DLod(uniform_sampler1D, float(0), float(0))
        && vec4(0) == texture1DProjLod(uniform_sampler1D, vec2(0), float(0))
        && vec4(0) == texture1DProjLod(uniform_sampler1D, vec4(0), float(0))
        && vec4(0) == texture2D(uniform_sampler2D, vec2(0))
        && vec4(0) == texture2D(uniform_sampler2D, vec2(0), float(0))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec3(0))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec3(0), float(0))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec4(0))
        && vec4(0) == texture2DProj(uniform_sampler2D, vec4(0), float(0))
        && vec4(0) == texture2DLod(uniform_sampler2D, vec2(0), float(0))
        && vec4(0) == texture2DProjLod(uniform_sampler2D, vec3(0), float(0))
        && vec4(0) == texture2DProjLod(uniform_sampler2D, vec4(0), float(0))
        && vec4(0) == texture3D(uniform_sampler3D, vec3(0))
        && vec4(0) == texture3D(uniform_sampler3D, vec3(0), float(0))
        && vec4(0) == texture3DProj(uniform_sampler3D, vec4(0))
        && vec4(0) == texture3DProj(uniform_sampler3D, vec4(0), float(0))
        && vec4(0) == texture3DLod(uniform_sampler3D, vec3(0), float(0))
        && vec4(0) == texture3DProjLod(uniform_sampler3D, vec4(0), float(0))
        && vec4(0) == textureCube(uniform_samplerCube, vec3(0))
        && vec4(0) == textureCube(uniform_samplerCube, vec3(0), float(0))
        && vec4(0) == textureCubeLod(uniform_samplerCube, vec3(0), float(0))
        && vec4(0) == shadow1D(uniform_sampler1DShadow, vec3(0))
        && vec4(0) == shadow1D(uniform_sampler1DShadow, vec3(0), float(0))
        && vec4(0) == shadow2D(uniform_sampler2DShadow, vec3(0))
        && vec4(0) == shadow2D(uniform_sampler2DShadow, vec3(0), float(0))
        && vec4(0) == shadow1DProj(uniform_sampler1DShadow, vec4(0))
        && vec4(0) == shadow1DProj(uniform_sampler1DShadow, vec4(0), float(0))
        && vec4(0) == shadow2DProj(uniform_sampler2DShadow, vec4(0))
        && vec4(0) == shadow2DProj(uniform_sampler2DShadow, vec4(0), float(0))
        && vec4(0) == shadow1DLod(uniform_sampler1DShadow, vec3(0), float(0))
        && vec4(0) == shadow2DLod(uniform_sampler2DShadow, vec3(0), float(0))
        && vec4(0) == shadow1DProjLod(uniform_sampler1DShadow, vec4(0), float(0))
        && vec4(0) == shadow2DProjLod(uniform_sampler2DShadow, vec4(0), float(0))
        ;
}

__generic<T : ITexelElement>
bool itextureFuncs(Sampler1D<T> gsampler1D
    , Sampler2D<T> gsampler2D
    , Sampler2DRect<T> gsampler2DRect
    , Sampler3D<T> gsampler3D
    , SamplerCube<T> gsamplerCube
    , Sampler1DArray<T> gsampler1DArray
    , Sampler2DArray<T> gsampler2DArray
    , SamplerCubeArray<T> gsamplerCubeArray
    , SamplerBuffer<T> gsamplerBuffer
)
{
    typealias gvec4 = vector<T.Element, 4>;

    constexpr ivec2 ivec2_0 = ivec2(0);

    return true
        // 8.9.1. Texture Query Functions
        && int(0) == textureSize(gsampler1D, int(0))
        && ivec2(0) == textureSize(gsampler2D, int(0))
        && ivec3(0) == textureSize(gsampler3D, int(0))
        && ivec2(0) == textureSize(gsamplerCube, int(0))
        && int(0) == textureSize(uniform_sampler1DShadow, int(0))
        && ivec2(0) == textureSize(uniform_sampler2DShadow, int(0))
        && ivec2(0) == textureSize(uniform_samplerCubeShadow, int(0))
        && ivec3(0) == textureSize(gsamplerCubeArray, int(0))
        && ivec3(0) == textureSize(uniform_samplerCubeArrayShadow, int(0))
        && ivec2(0) == textureSize(gsampler2DRect)
        && ivec2(0) == textureSize(uniform_sampler2DRectShadow)
        && ivec2(0) == textureSize(gsampler1DArray, int(0))
        && ivec2(0) == textureSize(uniform_sampler1DArrayShadow, int(0))
        && ivec3(0) == textureSize(gsampler2DArray, int(0))
        && ivec3(0) == textureSize(uniform_sampler2DArrayShadow, int(0))
        && int(0) == textureSize(gsamplerBuffer)

        && gvec4(T.Element(0)) == texelFetch(gsampler1D, int(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler2D, ivec2(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler3D, ivec3(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler2DRect, ivec2(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler1DArray, ivec2(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler2DArray, ivec3(0), int(0))
        && gvec4(T.Element(0)) == texelFetch(gsamplerBuffer, int(0))

        && gvec4(T.Element(0)) == texelFetchOffset(gsampler1D, int(0), int(0), __LINE__)
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2D, ivec2(0), int(0), { __LINE__ })
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler3D, ivec3(0), int(0), { __LINE__ })
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2DRect, ivec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler1DArray, ivec2(0), int(0), __LINE__)
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler2DArray, ivec3(0), int(0), { __LINE__ })
        // 8.9.4. Texture Gather Functions
        && gvec4(T.Element(0)) == textureGather(gsampler2D, vec2(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2D, vec2(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DArray, vec3(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DArray, vec3(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCube, vec3(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCube, vec3(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCubeArray, vec4(0))
        && gvec4(T.Element(0)) == textureGather(gsamplerCubeArray, vec4(0), int(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DRect, vec2(0))
        && gvec4(T.Element(0)) == textureGather(gsampler2DRect, vec2(0), int(0))
        && vec4(0) == textureGather(uniform_sampler2DShadow, vec2(0), float(0))
        && vec4(0) == textureGather(uniform_sampler2DArrayShadow, vec3(0), float(0))
        && vec4(0) == textureGather(uniform_samplerCubeShadow, vec3(0), float(0))
        && vec4(0) == textureGather(uniform_samplerCubeArrayShadow, vec4(0), float(0))
        && vec4(0) == textureGather(uniform_sampler2DRectShadow, vec2(0), float(0))
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2D, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2D, vec2(0), { __LINE__ }, int(0))
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DArray, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DArray, vec3(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffset(uniform_sampler2DShadow, vec2(0), float(0), { __LINE__ })
        && vec4(0) == textureGatherOffset(uniform_sampler2DArrayShadow, vec3(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DRect, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffset(gsampler2DRect, vec2(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffset(uniform_sampler2DRectShadow, vec2(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2D, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2D, vec2(0), { __LINE__ }, int(0))
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DArray, vec3(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DArray, vec3(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffsets(uniform_sampler2DShadow, vec2(0), float(0), { __LINE__ })
        && vec4(0) == textureGatherOffsets(uniform_sampler2DArrayShadow, vec3(0), float(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DRect, vec2(0), { __LINE__ })
        && gvec4(T.Element(0)) == textureGatherOffsets(gsampler2DRect, vec2(0), { __LINE__ }, int(0))
        && vec4(0) == textureGatherOffsets(uniform_sampler2DRectShadow, vec2(0), float(0), { __LINE__ })
        ;
}

#ifdef COMPUTE
[numthreads(2, 2, 1)]
#endif
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK_GLSL: void main(
    // CHECK_SPV: OpEntryPoint
    // CHECK_SPV_DIRECT: OpEntryPoint
    // CHECK_HLSL: void computeMain(
    // CHECK_CUDA: void computeMain(
    // CHECK_CPP: void _computeMain(

    outputBuffer.result = true
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
            )
        && itextureFuncs(
            uniform_isampler1D
            , uniform_isampler2D
            , uniform_isampler2DRect
            , uniform_isampler3D
            , uniform_isamplerCube
            , uniform_isampler1DArray
            , uniform_isampler2DArray
            , uniform_isamplerCubeArray
            , uniform_isamplerBuffer
            )
        && itextureFuncs(
            uniform_usampler1D
            , uniform_usampler2D
            , uniform_usampler2DRect
            , uniform_usampler3D
            , uniform_usamplerCube
            , uniform_usampler1DArray
            , uniform_usampler2DArray
            , uniform_usamplerCubeArray
            , uniform_usamplerBuffer
            );

    //outputBuffer[0] = float(r);
    //o_color[0] = float(r);
}

