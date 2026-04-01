#version 450 core
#extension GL_EXT_texture_shadow_lod : enable

//TEST:SIMPLE(filecheck=CHECK_GLSL_QUAD_C): -allow-glsl -stage compute -entry computeMain -target glsl -DCOMPUTE -DQUAD
// CHECK_GLSL_QUAD_C: GL_NV_compute_shader_derivatives
// CHECK_GLSL_QUAD_C: layout(derivative_group_quadsNV)

//TEST:SIMPLE(filecheck=CHECK_GLSL_LINEAR_C): -allow-glsl -stage compute -entry computeMain -target glsl -DCOMPUTE -DLINEAR
// CHECK_GLSL_LINEAR_C: GL_NV_compute_shader_derivatives
// CHECK_GLSL_LINEAR_C: layout(derivative_group_linearNV)

//TEST:SIMPLE(filecheck=CHECK_SPV_QUAD_C): -allow-glsl -stage compute -entry computeMain -target spirv -DQUAD -DCOMPUTE
// CHECK_SPV_QUAD_C: DerivativeGroupQuads{{NV|KHR}}
// CHECK_SPV_QUAD_C-NOT: DerivativeGroupQuads{{NV|KHR}}
// CHECK_SPV_QUAD_C: "SPV_KHR_compute_shader_derivatives"
// CHECK_SPV_QUAD_C-NOT: "SPV_KHR_compute_shader_derivatives"

//TEST:SIMPLE(filecheck=CHECK_SPV_LINEAR_C): -allow-glsl -stage compute -entry computeMain -target spirv -DLINEAR -DCOMPUTE
// CHECK_SPV_LINEAR_C: DerivativeGroupLinear{{NV|KHR}}
// CHECK_SPV_LINEAR_C-NOT: DerivativeGroupLinear{{NV|KHR}}
// CHECK_SPV_LINEAR_C: "SPV_KHR_compute_shader_derivatives"
// CHECK_SPV_LINEAR_C-NOT: "SPV_KHR_compute_shader_derivatives"

//TEST:SIMPLE(filecheck=CHECK_HLSL_C): -allow-glsl -stage compute -entry computeMain -target hlsl -DCOMPUTE
// CHECK_HLSL_C: computeMain(

//TEST:SIMPLE(filecheck=CHECK_GLSL_F): -allow-glsl -stage fragment -entry computeMain -target glsl 
// CHECK_GLSL_F-NOT: GL_NV_compute_shader_derivatives

//TEST:SIMPLE(filecheck=CHECK_SPV_QUAD_F): -allow-glsl -stage fragment -entry computeMain -target spirv -DQUAD
// CHECK_SPV_QUAD_F-NOT: DerivativeGroupQuads{{NV|KHR}}
// CHECK_SPV_QUAD_F-NOT: "SPV_KHR_compute_shader_derivatives"

//TEST:SIMPLE(filecheck=CHECK_SPV_LINEAR_F): -allow-glsl -stage fragment -entry computeMain -target spirv -DLINEAR
// CHECK_SPV_LINEAR_F-NOT: DerivativeGroupLinear{{NV|KHR}}
// CHECK_SPV_LINEAR_F-NOT: "SPV_KHR_compute_shader_derivatives"

//TEST:SIMPLE(filecheck=CHECK_GLSL_QUAD_F): -allow-glsl -stage fragment -entry computeMain -target glsl
// CHECK_GLSL_QUAD_F-NOT: GL_NV_compute_shader_derivatives
// CHECK_GLSL_QUAD_F-NOT: layout(derivative_group_quadsNV)

//TEST:SIMPLE(filecheck=CHECK_HLSL_F): -allow-glsl -stage fragment -entry computeMain -target hlsl
// CHECK_HLSL_F: computeMain(

buffer MyBlockName
{
    int result;
} outputBuffer;

uniform sampler1D uniform_sampler1D;

__generic<T : ITexelElement>
bool textureFuncs(Sampler1D<T> gsampler1D)
{
    typealias gvec4 = vector<T.Element,4>;

    constexpr ivec2 ivec2_0 = ivec2(0);

    return true
        // 8.9.1. Texture Query Functions
        && int(0) == textureSize(gsampler1D, int(0))
        && vec2(0) == textureQueryLod(gsampler1D, float(0))
        && int(0) == textureQueryLevels(gsampler1D)
        && gvec4(T.Element(0)) == texture(gsampler1D, float(0))
        && gvec4(T.Element(0)) == texture(gsampler1D, float(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec2(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec2(0), float(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec4(0))
        && gvec4(T.Element(0)) == textureProj(gsampler1D, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureLod(gsampler1D, float(0), float(0))
        && gvec4(T.Element(0)) == textureOffset(gsampler1D, float(0), __LINE__)
        && gvec4(T.Element(0)) == textureOffset(gsampler1D, float(0), __LINE__, float(0))
        && gvec4(T.Element(0)) == texelFetch(gsampler1D, int(0), int(0))
        && gvec4(T.Element(0)) == texelFetchOffset(gsampler1D, int(0), int(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec2(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec2(0), __LINE__, float(0))
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec4(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjOffset(gsampler1D, vec4(0), __LINE__,float(0))
        && gvec4(T.Element(0)) == textureLodOffset(gsampler1D, float(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjLod(gsampler1D, vec2(0), float(0))
        && gvec4(T.Element(0)) == textureProjLod(gsampler1D, vec4(0), float(0))
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler1D, vec2(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjLodOffset(gsampler1D, vec4(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureGrad(gsampler1D, float(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureGradOffset(gsampler1D, float(0), float(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjGrad(gsampler1D, vec2(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureProjGrad(gsampler1D, vec4(0), float(0), float(0))
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler1D, vec2(0), float(0), float(0), __LINE__)
        && gvec4(T.Element(0)) == textureProjGradOffset(gsampler1D, vec4(0), float(0), float(0), __LINE__)
        && vec4(0) == texture1D(uniform_sampler1D, float(0))
        && vec4(0) == texture1D(uniform_sampler1D, float(0), float(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec2(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec2(0), float(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec4(0))
        && vec4(0) == texture1DProj(uniform_sampler1D, vec4(0), float(0))
        && vec4(0) == texture1DLod(uniform_sampler1D, float(0), float(0))
        && vec4(0) == texture1DProjLod(uniform_sampler1D, vec2(0), float(0))
        && vec4(0) == texture1DProjLod(uniform_sampler1D, vec4(0), float(0))
        ;
}

bool derivativeFunc()
{
    return true
        && dFdx(1.0f) != -1.0f
        && dFdy(1.0f) != -1.0f
        && dFdxFine(1.0f) != -1.0f
        && dFdyFine(1.0f) != -1.0f
        && dFdxCoarse(1.0f) != -1.0f
        && dFdyCoarse(1.0f) != -1.0f
        && fwidth(1.0f) != -1.0f
        && fwidthFine(1.0f) != -1.0f
        && fwidthCoarse(1.0f) != -1.0f
        ;
}


#ifdef QUAD
[DerivativeGroupQuad]
#endif
#ifdef LINEAR
[DerivativeGroupLinear]
#endif
#ifdef COMPUTE
[numthreads(2, 2, 1)]
#endif
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK_SPV: OpEntryPoint
    outputBuffer.result = true
        && textureFuncs(uniform_sampler1D)
        && derivativeFunc()
        ;
}
