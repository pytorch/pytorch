//DISABLE_TEST:SIMPLE:-target dxil-assembly -entry fragmentMain -profile sm_6_5 -stage fragment -DNV_SHADER_EXTN_SLOT=u0 

//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry fragmentMain -stage fragment
//TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly -entry fragmentMain -profile sm_6_5 -stage fragment -DNV_SHADER_EXTN_SLOT=u0 
//TEST:SIMPLE(filecheck=HLSL):-target hlsl -entry fragmentMain -stage fragment

//DISABLED_TEST:SIMPLE:-target spirv-assembly -entry fragmentMain -stage fragment
//DISABLED_TEST:SIMPLE:-target dxil-assembly -entry fragmentMain -stage fragment
//DISABLED_TEST:SIMPLE:-target hlsl -entry fragmentMain -stage fragment

uniform Texture2D textures[] : register(t2, space10);
uniform SamplerState sampler;
uniform RWStructuredBuffer<uint> outputBuffer;

static Texture2D _getBindlessTexture2d(uint texIdx)
{
    return textures[NonUniformResourceIndex(texIdx)];
}

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

    accumulate(r, _getBindlessTexture2d(index).queryFootprintCoarse(granularity, sampler, coords));

// SPIRV: Extension  "SPV_NV_shader_image_footprint"
// SPIRV: ImageSampleFootprintNV

// DXIL: struct struct.NvShaderExtnStruct

// HLSL: NvFootprintCoarse

    outputBuffer[index] = r;
}

