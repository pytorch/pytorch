//TEST:SIMPLE(filecheck=CHECK0):-target glsl -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK1):-target hlsl -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK2):-target spirv -entry main -stage compute -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=CHECK3):-target spirv -entry main -stage compute -emit-spirv-directly

RWTexture2D<uint> texArray[2];

[numthreads(1,1,1)]
void main( uint2 dispatchThreadID : SV_DispatchThreadID, uint2 groupThreadID : SV_GroupThreadID )
{

    // CHECK0: {{.*}}imageAtomicAdd(texArray_{{.*}}[nonuniformEXT({{.*}})]

    // CHECK1: InterlockedAdd(texArray_{{.*}}[NonUniformResourceIndex({{.*}})]
    
    // CHECK2-DAG: OpDecorate %[[VAR1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2-DAG: OpDecorate %[[VAR2:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2-DAG: OpDecorate %[[VAR3:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2: %[[VAR1]] = OpCopyObject %{{.*}}
    // CHECK2: %[[VAR2]] = OpAccessChain %_ptr_UniformConstant_{{.*}} %{{.*}} %[[VAR1]]
    // CHECK2: %[[VAR3]] = OpImageTexelPointer %_ptr_Image_uint %[[VAR2]] %{{.*}}

    // CHECK3-DAG: OpDecorate %[[VAR1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK3: %[[VAR1]] = OpAccessChain %_ptr_UniformConstant_{{.*}} %texArray %{{.*}}
    // CHECK3: %{{.*}} = OpImageTexelPointer %_ptr_Image_uint %[[VAR1]] %{{.*}}
    InterlockedAdd(texArray[NonUniformResourceIndex(dispatchThreadID.x)][uint2(0)], 2);
}
