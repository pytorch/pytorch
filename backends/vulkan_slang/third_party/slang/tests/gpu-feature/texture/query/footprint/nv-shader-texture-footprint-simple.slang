//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry fragmentMain -stage fragment -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry fragmentMain -stage fragment
//DISABLED_TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry fragmentMain -stage fragment -DENABLE_CLAMP -emit-spirv-directly
//DISABLED_TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly -entry fragmentMain -stage fragment
//TEST:SIMPLE(filecheck=HLSL):-target hlsl -entry fragmentMain -stage fragment

// TODO: The SPIRV test is fine locally, but seems to lead to an impossible to debug hang in CI

//DISABLED_TEST:SIMPLE:-target spirv-assembly -entry fragmentMain -stage fragment
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

void accumulate(inout uint r, TextureFootprint2D f)
{
    accumulate(r, f.anchor);
    accumulate(r, f.offset);
    accumulate(r, f.mask);
    accumulate(r, f.lod);
    accumulate(r, f.granularity);
    accumulate(r, f.isSingleLevel);
}

uniform Texture2D texture;
uniform SamplerState sampler;
uniform RWStructuredBuffer<uint> outputBuffer;

cbuffer Uniforms
{
    uniform float2 coords;
    uniform uint granularity;
};

void fragmentMain(
    float v : VARYING)
{
    uint index = uint(v);
    uint r = 0;

    accumulate(r, texture.queryFootprintCoarse(granularity, sampler, coords));

// SPIRV: OpExtension "SPV_NV_shader_image_footprint"
// SPIRV: OpImageSampleFootprintNV

// HLSL: NvFootprintCoarse

    outputBuffer[index] = r;
}

