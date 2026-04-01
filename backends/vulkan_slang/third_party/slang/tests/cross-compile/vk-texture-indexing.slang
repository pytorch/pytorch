// vk-texture-indexing.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment
//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment -emit-spirv-directly

struct Params
{
    Texture2D<float> textures[10];
    SamplerState sampler;
};

ParameterBlock<Params> gParams;

float fetchData(uint2 coords, uint index)
{
    return gParams.textures[NonUniformResourceIndex(index)][coords];
}

float4 main(
	uint3 uv : UV)
    : SV_Target
{
    // CHECK: OpDecorate %{{.*}} NonUniform
    // CHECK: OpImageFetch
    float v = fetchData(uv.xy, uv.z);
    return v;
}
