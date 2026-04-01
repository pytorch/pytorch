//TEST:SIMPLE(filecheck=CHECK0):-target glsl -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK1):-target hlsl -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK2):-target spirv -entry main -stage compute -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=CHECK3):-target spirv -entry main -stage compute -emit-spirv-directly

RWStructuredBuffer<int> buffer[];

[numthreads(8, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // CHECK0: buffer_{{.*}}[nonuniformEXT({{.*}})]

    // CHECK1: buffer_{{.*}}[NonUniformResourceIndex(_{{.*}})]

    // CHECK2-DAG: OpDecorate %[[VAR1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2-DAG: OpDecorate %[[VAR2:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2-DAG: OpDecorate %[[VAR3:[a-zA-Z0-9_]+]] NonUniform
    // CHECK2: %[[VAR1]] = OpCopyObject %{{.*}}
    // CHECK2: %[[VAR2]] = OpAccessChain %_ptr_Uniform_int %buffer_{{.*}} %[[VAR1]]
    // CHECK2: OpStore  %[[VAR2]] %{{.*}}
    // CHECK2: %[[VAR3]] = OpCopyObject %{{.*}}
    // CHECK2: %{{.*}} = OpBitcast %int %[[VAR3]]

    // CHECK3: %[[VAR1:[a-zA-Z0-9_]+]] = OpAccessChain %_ptr_StorageBuffer_RWStructuredBuffer %buffer %{{.*}}NonUniform
    // CHECK3: %{{.*}} = OpAccessChain %_ptr_StorageBuffer_int %[[VAR1]]
    // CHECK3: %[[VAR2:[a-zA-Z0-9_]+]] = OpAccessChain %_ptr_StorageBuffer_RWStructuredBuffer %buffer %{{.*}}NonUniform
    // CHECK3: %{{.*}} = OpAccessChain %_ptr_StorageBuffer_int %[[VAR2]]
    RWStructuredBuffer<int> buffer1 = buffer[NonUniformResourceIndex(NonUniformResourceIndex(NonUniformResourceIndex(dispatchThreadID.x)))];
    buffer1[0] = 1;

    RWStructuredBuffer<int> buffer2 = buffer[int(NonUniformResourceIndex(uint(float(dispatchThreadID.x))))];
    buffer2[0] = 1;
}