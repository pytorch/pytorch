//TEST:SIMPLE(filecheck=CHECK0):-target glsl -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK1):-target hlsl -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK2):-target spirv -entry main -stage compute -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=CHECK3):-target spirv -entry main -stage compute -emit-spirv-directly

Texture2D<float> textures[2];
RWTexture2D<float2> outputTexture;
[numthreads(1, 1, 1)]
void main(uint2 pixelIndex : SV_DispatchThreadID)
{
    // CHECK0: texelFetch({{.*}}[nonuniformEXT({{.*}})])

    // CHECK1: [NonUniformResourceIndex(_{{.*}})].Load(_{{.*}})

    // CHECK2: OpDecorate %[[VAR1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2: OpDecorate %[[VAR2:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2: OpDecorate %[[VAR3:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2: OpDecorate %[[VAR4:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2: OpDecorate %[[VAR5:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2: OpDecorate %[[VAR6:[a-zA-Z0-9_]+]] NonUniform

    // CHECK2: %[[VAR1]] = OpCopyObject %{{.*}}
    // CHECK2: %[[VAR2]] = OpAccessChain %_ptr_UniformConstant_{{.*}} %{{.*}} %[[VAR1]]
    // CHECK2: %[[VAR3]] = OpLoad %{{.*}} %[[VAR2]]
    // CHECK2: %[[VAR4]] = OpCopyObject %{{.*}}
    // CHECK2: %[[VAR5]] = OpAccessChain %_ptr_UniformConstant_{{.*}} %{{.*}} %[[VAR4]]
    // CHECK2: %[[VAR6]] = OpLoad %{{.*}} %[[VAR5]]

    // CHECK3: %[[VAR1:[a-zA-Z0-9_]+]] = OpAccessChain{{.*}}NonUniform
    // CHECK3: %[[VAR2:[a-zA-Z0-9_]+]] = OpLoad %{{.*}}[[VAR1]]{{.*}}NonUniform
    // CHECK3: %{{.*}} = OpImageFetch %v4float %[[VAR2]] %{{.*}}
    // CHECK3: %[[VAR3:[a-zA-Z0-9_]+]] = OpAccessChain{{.*}}NonUniform
    // CHECK3: %[[VAR4:[a-zA-Z0-9_]+]] = OpLoad %{{.*}}[[VAR3]]{{.*}}NonUniform
    // CHECK3: %{{.*}} = OpImageFetch %v4float %[[VAR4]] %{{.*}}
    float2 tmp0 = textures[NonUniformResourceIndex(pixelIndex.x)].Load(int3(0, 0, 0));
    outputTexture[0] = tmp0;

    float2 tmp1 = textures[NonUniformResourceIndex(uint(float(pixelIndex.x)))].Load(int3(0, 0, 0));
    outputTexture[1] = tmp1;
}

