//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -emit-spirv-via-glsl -entry fragmentMain -stage fragment -DENABLE_CLAMP
//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry fragmentMain -stage fragment -DENABLE_CLAMP
//DISABLED_TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry fragmentMain -stage fragment -DENABLE_CLAMP -emit-spirv-directly

//DISABLED_TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly -entry fragmentMain -stage fragment
//TEST:SIMPLE(filecheck=HLSL):-target hlsl -entry fragmentMain -stage fragment

// TODO: The SPIRV test is fine locally, but seems to lead to an impossible to debug hang in CI

//DISABLED_TEST:SIMPLE:-target spirv-assembly -entry fragmentMain -stage fragment -DENABLE_CLAMP
//DISABLED_TEST:SIMPLE:-target dxil-assembly -entry fragmentMain -stage fragment
//DISABLED_TEST:SIMPLE:-target hlsl -entry fragmentMain -stage fragment

// In order to make sure that all of the data produced in each
// query is referenced (and thus cannot be DCE'd away), we maintain
// single accumulator that will have the bits of each query result
// mixed into it.

void accumulate(inout uint r, uint u)
{
    r = r ^ u;
}

void accumulate(inout uint r, bool b)
{
    accumulate(r, uint(b));
}

void accumulate(inout uint r, uint2 u)
{
    accumulate(r, u.x);
    accumulate(r, u.y);
}

void accumulate(inout uint r, uint3 u)
{
    accumulate(r, u.x);
    accumulate(r, u.y);
    accumulate(r, u.z);
}

void accumulate_2D(inout uint r, TextureFootprint2D f)
{
    accumulate(r, f.anchor);
// HLSL-DAG: NvFootprintExtractAnchorTileLoc2D

    accumulate(r, f.offset);
// HLSL-DAG: NvFootprintExtractOffset2D

    accumulate(r, f.mask);
// HLSL-DAG: NvFootprintExtractBitmask

    accumulate(r, f.lod);
// HLSL-DAG: NvFootprintExtractLOD

    accumulate(r, f.granularity);
// HLSL-DAG: NvFootprintExtractReturnGran

    accumulate(r, f.isSingleLevel);
}

void accumulate_3D(inout uint r, TextureFootprint3D f)
{
    accumulate(r, f.anchor);
// HLSL-DAG: NvFootprintExtractAnchorTileLoc3D

    accumulate(r, f.offset);
// HLSL-DAG: NvFootprintExtractOffset3D

    accumulate(r, f.mask);
// HLSL-DAG: NvFootprintExtractBitmask

    accumulate(r, f.lod);
// HLSL-DAG: NvFootprintExtractLOD

    accumulate(r, f.granularity);
// HLSL-DAG: NvFootprintExtractReturnGran

    accumulate(r, f.isSingleLevel);
}

// SPRIV: OpExtension "SPV_NV_shader_image_footprint"

void fragmentMain(
            float                       v : VARYING,

    uniform Texture2D                   texture_2D,
// SPIRV-DAG: OpTypeImage {{.*}} 2D

// HLSL-DAG: Texture2D{{.*}}register(t0)

    uniform Texture3D                   texture_3D,
// SPIRV-DAG: OpTypeImage {{.*}} 3D

// HLSL-DAG: Texture3D{{.*}}register(t1)

    uniform SamplerState                sampler,
    uniform RWStructuredBuffer<uint>    outputBuffer,
    uniform float4                      c,
    uniform float4                      gx,
    uniform float4                      gy,
    uniform uint                        g,
    uniform float                       lodBias,
    uniform float                       lodClamp,
    uniform float                       lod)
{
    uint index = uint(v);
    uint r = 0;

    TextureFootprint2D.Granularity granularity = g;

#ifdef ENABLE_CLAMP
#define CLAMP(ARG) ARG
#else
#define CLAMP(ARG) /* empty */
#endif

#define TEST_COMMON(ND, COARSE_OR_FINE) \
        accumulate##ND(r, texture##ND.queryFootprint##COARSE_OR_FINE            (granularity, sampler, coords                   ))  ;\
        accumulate##ND(r, texture##ND.queryFootprint##COARSE_OR_FINE##Bias      (granularity, sampler, coords, lodBias          ))  ;\
CLAMP(  accumulate##ND(r, texture##ND.queryFootprint##COARSE_OR_FINE##Clamp     (granularity, sampler, coords, lodClamp         ))) ;\
CLAMP(  accumulate##ND(r, texture##ND.queryFootprint##COARSE_OR_FINE##BiasClamp (granularity, sampler, coords, lodBias, lodClamp))) ;\
        accumulate##ND(r, texture##ND.queryFootprint##COARSE_OR_FINE##Level     (granularity, sampler, coords, lod              ))  ;\
/* end */

#define TEST_GRAD(ND, COARSE_OR_FINE) \
        accumulate##ND(r, texture##ND.queryFootprint##COARSE_OR_FINE##Grad      (granularity, sampler, coords, dx, dy           ))  ;\
CLAMP(  accumulate##ND(r, texture##ND.queryFootprint##COARSE_OR_FINE##GradClamp (granularity, sampler, coords, dx, dy, lodClamp ))) ;\
/* end */


#define TEST_2D(COARSE_OR_FINE) \
    TEST_COMMON(_2D, COARSE_OR_FINE) \
    TEST_GRAD(_2D, COARSE_OR_FINE) \
    /* end */

#define TEST_3D(COARSE_OR_FINE) \
    TEST_COMMON(_3D, COARSE_OR_FINE) \
    /* end */

// SPIRV-DAG: [[TRUE:%[a-zA-Z0-9_]+]] = OpConstantTrue
// SPIRV-DAG: [[FALSE:%[a-zA-Z0-9_]+]] = OpConstantFalse

    {
        let coords = c.xy;
        let dx = gx.xy;
        let dy = gy.xy;

        TEST_2D(Coarse);

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]]
// HLSL-DAG: NvFootprintCoarse{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Bias
// HLSL-DAG: NvFootprintCoarseBias{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Bias|MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Lod
// HLSL-DAG: NvFootprintCoarseLevel{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Grad
// HLSL-DAG: NvFootprintCoarseGrad{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Grad|MinLod

        TEST_2D(Fine);

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]]
// HLSL-DAG: NvFootprintFine{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Bias
// HLSL-DAG: NvFootprintFineBias{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Bias|MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Lod
// HLSL-DAG: NvFootprintFineLevel{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Grad
// HLSL-DAG: NvFootprintFineGrad{{.*}}NV_EXTN_TEXTURE_2D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Grad|MinLod

    }

    {
        let coords = c.xyz;
        let dx = gx.xyz;
        let dy = gy.xyz;

        TEST_3D(Coarse);

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]]
// HLSL-DAG: NvFootprintCoarse{{.*}}NV_EXTN_TEXTURE_3D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Bias
// HLSL-DAG: NvFootprintCoarseBias{{.*}}NV_EXTN_TEXTURE_3D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Bias|MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[TRUE]] Lod
// HLSL-DAG: NvFootprintCoarseLevel{{.*}}NV_EXTN_TEXTURE_3D

        TEST_3D(Fine);

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]]
// HLSL-DAG: NvFootprintFine{{.*}}NV_EXTN_TEXTURE_3D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Bias
// HLSL-DAG: NvFootprintFineBias{{.*}}NV_EXTN_TEXTURE_3D

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Bias|MinLod

// SPIRV: OpImageSampleFootprintNV
// SPIRV-SAME: [[FALSE]] Lod
// HLSL-DAG: NvFootprintFineLevel{{.*}}NV_EXTN_TEXTURE_3D

    }

    outputBuffer[index] = r;
}
