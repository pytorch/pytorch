//

// Define all the various language "profiles" we want to support.

#ifndef LANGUAGE
#define LANGUAGE(TAG, NAME) /* empty */
#endif

#ifndef LANGUAGE_ALIAS
#define LANGUAGE_ALIAS(TAG, NAME) /* empty */
#endif

#ifndef PROFILE_FAMILY
#define PROFILE_FAMILY(TAG) /* empty */
#endif

#ifndef PROFILE_VERSION
#define PROFILE_VERSION(TAG, FAMILY) /* empty */
#endif

#ifndef PROFILE_STAGE
#define PROFILE_STAGE(TAG, NAME, VAL) /* empty */
#endif

#ifndef PROFILE_STAGE_ALIAS
#define PROFILE_STAGE_ALIAS(TAG, NAME, VAL) /* empty */
#endif


#ifndef PROFILE
#define PROFILE(TAG, NAME, STAGE, VERSION) /* empty */
#endif

#ifndef PROFILE_ALIAS
#define PROFILE_ALIAS(TAG, DEF, NAME) /* empty */
#endif

// Source and destination languages

LANGUAGE(HLSL, hlsl)
LANGUAGE(DXBytecode, dxbc)
LANGUAGE(DXBytecodeAssembly, dxbc_asm)
LANGUAGE(DXIL, dxil)
LANGUAGE(DXILAssembly, dxil_asm)
LANGUAGE(GLSL, glsl)
LANGUAGE(GLSL_ES, glsl_es)
LANGUAGE(GLSL_VK, glsl_vk)
LANGUAGE(SPIRV, spirv)
LANGUAGE(SPIRV_GL, spirv_gl)
LANGUAGE(C, c)
LANGUAGE(CPP, cpp)
LANGUAGE(METAL, metal)
LANGUAGE_ALIAS(GLSL, glsl_gl)
LANGUAGE_ALIAS(SPIRV, spirv_vk)


// Pipeline stages to target
PROFILE_STAGE(Vertex, vertex, SLANG_STAGE_VERTEX)
PROFILE_STAGE(Hull, hull, SLANG_STAGE_HULL)
PROFILE_STAGE_ALIAS(TessControl, tesscontrol, Hull)
PROFILE_STAGE(Domain, domain, SLANG_STAGE_DOMAIN)
PROFILE_STAGE_ALIAS(TessEval, tesseval, Domain)
PROFILE_STAGE(Geometry, geometry, SLANG_STAGE_GEOMETRY)
PROFILE_STAGE(Pixel, pixel, SLANG_STAGE_FRAGMENT)
PROFILE_STAGE(Compute, compute, SLANG_STAGE_COMPUTE)

PROFILE_STAGE(RayGeneration, raygeneration, SLANG_STAGE_RAY_GENERATION)
PROFILE_STAGE(Intersection, intersection, SLANG_STAGE_INTERSECTION)
PROFILE_STAGE(AnyHit, anyhit, SLANG_STAGE_ANY_HIT)
PROFILE_STAGE(ClosestHit, closesthit, SLANG_STAGE_CLOSEST_HIT)
PROFILE_STAGE(Miss, miss, SLANG_STAGE_MISS)
PROFILE_STAGE(Callable, callable, SLANG_STAGE_CALLABLE)

PROFILE_STAGE(Mesh, mesh, SLANG_STAGE_MESH)
PROFILE_STAGE(Amplification, amplification, SLANG_STAGE_AMPLIFICATION)
PROFILE_STAGE(Dispatch, dispatch, SLANG_STAGE_DISPATCH)

// Note: HLSL and Direct3D convention erroneously uses the term "Pixel Shader"
// for the thing that shades *fragments*. Slang strives to treat the more correct
// term "Fragment Shader" as the primary one, but in order to be compatible with
// existing HLSL conventions, we need to treat `pixel` as the official stage
// name and `fragment` as an alias for it here, because the lower-case stage
// names are used to drive output HLSL generation.
//
PROFILE_STAGE_ALIAS(Fragment, fragment, Pixel)

// Profile families

PROFILE_FAMILY(DX)
PROFILE_FAMILY(GLSL)
PROFILE_FAMILY(METAL)
PROFILE_FAMILY(SPIRV)

// Profile versions
PROFILE_VERSION(DX_4_0, DX)
PROFILE_VERSION(DX_4_1, DX)
PROFILE_VERSION(DX_5_0, DX)
PROFILE_VERSION(DX_5_1, DX)
PROFILE_VERSION(DX_6_0, DX)
PROFILE_VERSION(DX_6_1, DX)
PROFILE_VERSION(DX_6_2, DX)
PROFILE_VERSION(DX_6_3, DX)
PROFILE_VERSION(DX_6_4, DX)
PROFILE_VERSION(DX_6_5, DX)
PROFILE_VERSION(DX_6_6, DX)
PROFILE_VERSION(DX_6_7, DX)
PROFILE_VERSION(DX_6_8, DX)
PROFILE_VERSION(DX_6_9, DX)

PROFILE_VERSION(GLSL_150, GLSL)
PROFILE_VERSION(GLSL_330, GLSL)
PROFILE_VERSION(GLSL_400, GLSL)
PROFILE_VERSION(GLSL_410, GLSL)
PROFILE_VERSION(GLSL_420, GLSL)
PROFILE_VERSION(GLSL_430, GLSL)
PROFILE_VERSION(GLSL_440, GLSL)
PROFILE_VERSION(GLSL_450, GLSL)
PROFILE_VERSION(GLSL_460, GLSL)

PROFILE_VERSION(METAL_2_3, METAL)
PROFILE_VERSION(METAL_2_4, METAL)

PROFILE_VERSION(SPIRV_1_0, SPIRV)
PROFILE_VERSION(SPIRV_1_1, SPIRV)
PROFILE_VERSION(SPIRV_1_2, SPIRV)
PROFILE_VERSION(SPIRV_1_3, SPIRV)
PROFILE_VERSION(SPIRV_1_4, SPIRV)
PROFILE_VERSION(SPIRV_1_5, SPIRV)
PROFILE_VERSION(SPIRV_1_6, SPIRV)

// Specific profiles
PROFILE(DX_Compute_4_0, cs_4_0, Compute, DX_4_0)
PROFILE(DX_Compute_4_1, cs_4_1, Compute, DX_4_1)
PROFILE(DX_Compute_5_0, cs_5_0, Compute, DX_5_0)
PROFILE(DX_Compute_5_1, cs_5_1, Compute, DX_5_1)
PROFILE(DX_Compute_6_0, cs_6_0, Compute, DX_6_0)
PROFILE(DX_Compute_6_1, cs_6_1, Compute, DX_6_1)
PROFILE(DX_Compute_6_2, cs_6_2, Compute, DX_6_2)
PROFILE(DX_Compute_6_3, cs_6_3, Compute, DX_6_3)
PROFILE(DX_Compute_6_4, cs_6_4, Compute, DX_6_4)
PROFILE(DX_Compute_6_5, cs_6_5, Compute, DX_6_5)
PROFILE(DX_Compute_6_6, cs_6_6, Compute, DX_6_6)
PROFILE(DX_Compute_6_7, cs_6_7, Compute, DX_6_7)
PROFILE(DX_Compute_6_8, cs_6_8, Compute, DX_6_8)
PROFILE(DX_Compute_6_9, cs_6_9, Compute, DX_6_9)

PROFILE(DX_Domain_5_0, ds_5_0, Domain, DX_5_0)
PROFILE(DX_Domain_5_1, ds_5_1, Domain, DX_5_1)
PROFILE(DX_Domain_6_0, ds_6_0, Domain, DX_6_0)
PROFILE(DX_Domain_6_1, ds_6_1, Domain, DX_6_1)
PROFILE(DX_Domain_6_2, ds_6_2, Domain, DX_6_2)
PROFILE(DX_Domain_6_3, ds_6_3, Domain, DX_6_3)
PROFILE(DX_Domain_6_4, ds_6_4, Domain, DX_6_4)
PROFILE(DX_Domain_6_5, ds_6_5, Domain, DX_6_5)
PROFILE(DX_Domain_6_6, ds_6_6, Domain, DX_6_6)
PROFILE(DX_Domain_6_7, ds_6_7, Domain, DX_6_7)
PROFILE(DX_Domain_6_8, ds_6_8, Domain, DX_6_8)
PROFILE(DX_Domain_6_9, ds_6_9, Domain, DX_6_9)

PROFILE(DX_Geometry_4_0, gs_4_0, Geometry, DX_4_0)
PROFILE(DX_Geometry_4_1, gs_4_1, Geometry, DX_4_1)
PROFILE(DX_Geometry_5_0, gs_5_0, Geometry, DX_5_0)
PROFILE(DX_Geometry_5_1, gs_5_1, Geometry, DX_5_1)
PROFILE(DX_Geometry_6_0, gs_6_0, Geometry, DX_6_0)
PROFILE(DX_Geometry_6_1, gs_6_1, Geometry, DX_6_1)
PROFILE(DX_Geometry_6_2, gs_6_2, Geometry, DX_6_2)
PROFILE(DX_Geometry_6_3, gs_6_3, Geometry, DX_6_3)
PROFILE(DX_Geometry_6_4, gs_6_4, Geometry, DX_6_4)
PROFILE(DX_Geometry_6_5, gs_6_5, Geometry, DX_6_5)
PROFILE(DX_Geometry_6_6, gs_6_6, Geometry, DX_6_6)
PROFILE(DX_Geometry_6_7, gs_6_7, Geometry, DX_6_7)
PROFILE(DX_Geometry_6_8, gs_6_8, Geometry, DX_6_8)
PROFILE(DX_Geometry_6_9, gs_6_9, Geometry, DX_6_9)

PROFILE(DX_Hull_5_0, hs_5_0, Hull, DX_5_0)
PROFILE(DX_Hull_5_1, hs_5_1, Hull, DX_5_1)
PROFILE(DX_Hull_6_0, hs_6_0, Hull, DX_6_0)
PROFILE(DX_Hull_6_1, hs_6_1, Hull, DX_6_1)
PROFILE(DX_Hull_6_2, hs_6_2, Hull, DX_6_2)
PROFILE(DX_Hull_6_3, hs_6_3, Hull, DX_6_3)
PROFILE(DX_Hull_6_4, hs_6_4, Hull, DX_6_4)
PROFILE(DX_Hull_6_5, hs_6_5, Hull, DX_6_5)
PROFILE(DX_Hull_6_6, hs_6_6, Hull, DX_6_6)
PROFILE(DX_Hull_6_7, hs_6_7, Hull, DX_6_7)
PROFILE(DX_Hull_6_8, hs_6_8, Hull, DX_6_8)
PROFILE(DX_Hull_6_9, hs_6_9, Hull, DX_6_9)

PROFILE(DX_Fragment_4_0, ps_4_0, Fragment, DX_4_0)
PROFILE(DX_Fragment_4_1, ps_4_1, Fragment, DX_4_1)
PROFILE(DX_Fragment_5_0, ps_5_0, Fragment, DX_5_0)
PROFILE(DX_Fragment_5_1, ps_5_1, Fragment, DX_5_1)
PROFILE(DX_Fragment_6_0, ps_6_0, Fragment, DX_6_0)
PROFILE(DX_Fragment_6_1, ps_6_1, Fragment, DX_6_1)
PROFILE(DX_Fragment_6_2, ps_6_2, Fragment, DX_6_2)
PROFILE(DX_Fragment_6_3, ps_6_3, Fragment, DX_6_3)
PROFILE(DX_Fragment_6_4, ps_6_4, Fragment, DX_6_4)
PROFILE(DX_Fragment_6_5, ps_6_5, Fragment, DX_6_5)
PROFILE(DX_Fragment_6_6, ps_6_6, Fragment, DX_6_6)
PROFILE(DX_Fragment_6_7, ps_6_7, Fragment, DX_6_7)
PROFILE(DX_Fragment_6_8, ps_6_8, Fragment, DX_6_8)
PROFILE(DX_Fragment_6_9, ps_6_9, Fragment, DX_6_9)

PROFILE(DX_Vertex_4_0, vs_4_0, Vertex, DX_4_0)
PROFILE(DX_Vertex_4_1, vs_4_1, Vertex, DX_4_1)
PROFILE(DX_Vertex_5_0, vs_5_0, Vertex, DX_5_0)
PROFILE(DX_Vertex_5_1, vs_5_1, Vertex, DX_5_1)
PROFILE(DX_Vertex_6_0, vs_6_0, Vertex, DX_6_0)
PROFILE(DX_Vertex_6_1, vs_6_1, Vertex, DX_6_1)
PROFILE(DX_Vertex_6_2, vs_6_2, Vertex, DX_6_2)
PROFILE(DX_Vertex_6_3, vs_6_3, Vertex, DX_6_3)
PROFILE(DX_Vertex_6_4, vs_6_4, Vertex, DX_6_4)
PROFILE(DX_Vertex_6_5, vs_6_5, Vertex, DX_6_5)
PROFILE(DX_Vertex_6_6, vs_6_6, Vertex, DX_6_6)
PROFILE(DX_Vertex_6_7, vs_6_7, Vertex, DX_6_7)
PROFILE(DX_Vertex_6_8, vs_6_8, Vertex, DX_6_8)
PROFILE(DX_Vertex_6_9, vs_6_9, Vertex, DX_6_9)

PROFILE(DX_Mesh_6_5, ms_6_5, Mesh, DX_6_5)
PROFILE(DX_Mesh_6_6, ms_6_6, Mesh, DX_6_6)
PROFILE(DX_Mesh_6_7, ms_6_7, Mesh, DX_6_7)
PROFILE(DX_Mesh_6_8, ms_6_8, Mesh, DX_6_8)
PROFILE(DX_Mesh_6_9, ms_6_9, Mesh, DX_6_9)

PROFILE(DX_Amplification_6_5, as_6_5, Amplification, DX_6_5)
PROFILE(DX_Amplification_6_6, as_6_6, Amplification, DX_6_6)
PROFILE(DX_Amplification_6_7, as_6_7, Amplification, DX_6_7)
PROFILE(DX_Amplification_6_8, as_6_8, Amplification, DX_6_8)
PROFILE(DX_Amplification_6_9, as_6_9, Amplification, DX_6_9)

// TODO: consider making `lib_*_*` alias these...
PROFILE(DX_None_4_0, sm_4_0, Unknown, DX_4_0)
PROFILE(DX_None_4_1, sm_4_1, Unknown, DX_4_1)
PROFILE(DX_None_5_0, sm_5_0, Unknown, DX_5_0)
PROFILE(DX_None_5_1, sm_5_1, Unknown, DX_5_1)
PROFILE(DX_None_6_0, sm_6_0, Unknown, DX_6_0)

// From Shader Model 6.1 on, the dxc compiler recognizes a `lib` profile
// that can be used to compile multiple entry points. We want that
// `lib` name to be the default for how we render these profiles when
// invoking downstream tools, so we use that instead of the `sm_`
// prefix, and then re-introduce the `sm_` variants as aliases.
//
// TODO: We may eventually want a split between how Slang represents
// profiles and their names to users, vs. how it renders them when
// invoking downstream tools, so that the profile name in any
// error messages can be consistent with our `sm_*` naems above
//
PROFILE(DX_Lib_6_1, lib_6_1, Unknown, DX_6_1)
PROFILE(DX_Lib_6_2, lib_6_2, Unknown, DX_6_2)
PROFILE(DX_Lib_6_3, lib_6_3, Unknown, DX_6_3)
PROFILE(DX_Lib_6_4, lib_6_4, Unknown, DX_6_4)
PROFILE(DX_Lib_6_5, lib_6_5, Unknown, DX_6_5)
PROFILE(DX_Lib_6_6, lib_6_6, Unknown, DX_6_6)
PROFILE(DX_Lib_6_7, lib_6_7, Unknown, DX_6_7)
PROFILE(DX_Lib_6_8, lib_6_8, Unknown, DX_6_8)
PROFILE(DX_Lib_6_9, lib_6_9, Unknown, DX_6_9)

PROFILE_ALIAS(DX_None_6_1, DX_Lib_6_1, sm_6_1)
PROFILE_ALIAS(DX_None_6_2, DX_Lib_6_2, sm_6_2)
PROFILE_ALIAS(DX_None_6_3, DX_Lib_6_3, sm_6_3)
PROFILE_ALIAS(DX_None_6_4, DX_Lib_6_4, sm_6_4)
PROFILE_ALIAS(DX_None_6_5, DX_Lib_6_5, sm_6_5)
PROFILE_ALIAS(DX_None_6_6, DX_Lib_6_6, sm_6_6)
PROFILE_ALIAS(DX_None_6_7, DX_Lib_6_7, sm_6_7)
PROFILE_ALIAS(DX_None_6_8, DX_Lib_6_8, sm_6_8)
PROFILE_ALIAS(DX_None_6_9, DX_Lib_6_9, sm_6_9)

PROFILE(METAL_LIB_2_3, metallib_2_3, Unknown, METAL_2_3)
PROFILE(METAL_LIB_2_4, metallib_2_4, Unknown, METAL_2_4)

PROFILE(SPIRV_LIB_1_0, spirv_1_0, Unknown, SPIRV_1_0)
PROFILE(SPIRV_LIB_1_1, spirv_1_1, Unknown, SPIRV_1_1)
PROFILE(SPIRV_LIB_1_2, spirv_1_2, Unknown, SPIRV_1_2)
PROFILE(SPIRV_LIB_1_3, spirv_1_3, Unknown, SPIRV_1_3)
PROFILE(SPIRV_LIB_1_4, spirv_1_4, Unknown, SPIRV_1_4)
PROFILE(SPIRV_LIB_1_5, spirv_1_5, Unknown, SPIRV_1_5)
PROFILE(SPIRV_LIB_1_6, spirv_1_6, Unknown, SPIRV_1_6)

// Define all the GLSL profiles

PROFILE(GLSL_None_150, glsl_150, Unknown, GLSL_150)
PROFILE(GLSL_None_330, glsl_330, Unknown, GLSL_330)
PROFILE(GLSL_None_400, glsl_400, Unknown, GLSL_400)
PROFILE(GLSL_None_410, glsl_410, Unknown, GLSL_410)
PROFILE(GLSL_None_420, glsl_420, Unknown, GLSL_420)
PROFILE(GLSL_None_430, glsl_430, Unknown, GLSL_430)
PROFILE(GLSL_None_440, glsl_440, Unknown, GLSL_440)
PROFILE(GLSL_None_450, glsl_450, Unknown, GLSL_450)
PROFILE(GLSL_None_460, glsl_460, Unknown, GLSL_460)

#define P(UPPER, LOWER, VERSION) \
    PROFILE(GLSL_##UPPER##_##VERSION, glsl_##LOWER##_##VERSION, UPPER, GLSL_##VERSION)

P(Vertex, vertex, 150)
P(Vertex, vertex, 330)
P(Vertex, vertex, 400)
P(Vertex, vertex, 410)
P(Vertex, vertex, 420)
P(Vertex, vertex, 430)
P(Vertex, vertex, 440)
P(Vertex, vertex, 450)

P(Fragment, fragment, 150)
P(Fragment, fragment, 330)
P(Fragment, fragment, 400)
P(Fragment, fragment, 410)
P(Fragment, fragment, 420)
P(Fragment, fragment, 430)
P(Fragment, fragment, 440)
P(Fragment, fragment, 450)

P(Geometry, geometry, 150)
P(Geometry, geometry, 330)
P(Geometry, geometry, 400)
P(Geometry, geometry, 410)
P(Geometry, geometry, 420)
P(Geometry, geometry, 430)
P(Geometry, geometry, 440)
P(Geometry, geometry, 450)

P(Compute, compute, 430)
P(Compute, compute, 440)
P(Compute, compute, 450)

#undef P
#define P(UPPER, LOWER, STAGE, VERSION) \
    PROFILE(GLSL_##UPPER##_##VERSION, glsl_##LOWER##_##VERSION, STAGE, GLSL_##VERSION)

P(TessControl, tess_control, Hull, 400)
P(TessControl, tess_control, Hull, 410)
P(TessControl, tess_control, Hull, 420)
P(TessControl, tess_control, Hull, 430)
P(TessControl, tess_control, Hull, 440)
P(TessControl, tess_control, Hull, 450)

P(TessEval, tess_eval, Domain, 400)
P(TessEval, tess_eval, Domain, 410)
P(TessEval, tess_eval, Domain, 420)
P(TessEval, tess_eval, Domain, 430)
P(TessEval, tess_eval, Domain, 440)
P(TessEval, tess_eval, Domain, 450)

#undef P

// Define a default profile for each GLSL stage that just
// uses the latest language version we know of

PROFILE_ALIAS(GLSL_Vertex, GLSL_Vertex_450, glsl_vertex)
PROFILE_ALIAS(GLSL_Fragment, GLSL_Fragment_450, glsl_fragment)
PROFILE_ALIAS(GLSL_Geometry, GLSL_Geometry_450, glsl_geometry)
PROFILE_ALIAS(GLSL_TessControl, GLSL_TessControl_450, glsl_tess_control)
PROFILE_ALIAS(GLSL_TessEval, GLSL_TessEval_450, glsl_tess_eval)
PROFILE_ALIAS(GLSL_Compute, GLSL_Compute_450, glsl_compute)

// TODO: define a profile for each GLSL *version* that we can
// use as a catch-all when the stage can be inferred from
// something else

#undef LANGUAGE
#undef LANGUAGE_ALIAS
#undef PROFILE_FAMILY
#undef PROFILE_VERSION
#undef PROFILE_STAGE
#undef PROFILE_STAGE_ALIAS
#undef PROFILE
#undef PROFILE_ALIAS
