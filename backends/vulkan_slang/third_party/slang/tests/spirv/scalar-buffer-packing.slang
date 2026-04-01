// scalar-buffer-packing.slang
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute
//TEST:SIMPLE(filecheck=SPIRV): -stage compute -entry computeMain -target spirv -emit-spirv-directly -force-glsl-scalar-layout
// Test ability to directly output SPIR-V

//TEST_INPUT:set Uniforms.v0 = 1
//TEST_INPUT:set Uniforms.v1 = { 1.0, 2.0, 3.0}

cbuffer Uniforms
{
    bool v0;
    float3 v1[3];
}

struct Val
{
    bool x;
    int3 v1[2];
    [mutating]
    void set(bool v)
    {
        x = v;
        v1[0].x = 1;
        v1[0].y = 2;
        v1[0].z = 3;
        v1[1].x = 4;
        v1[1].y = 5;
        v1[1].z = 6;
    }
}

//TEST_INPUT:set result = out ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4)
RWStructuredBuffer<Val> result;
[numthreads(1,1,1)]
void computeMain()
{
    // CHECK: 1
    // CHECK: 0
    // CHECK: 0
    // CHECK: 0
    // CHECK: 1
    // CHECK: 2
    // CHECK: 3
    // CHECK: 0
    // CHECK: 4
    // CHECK: 5
    // CHECK: 6

    result[0].set(v0);
}

// SPIRV: OpEntryPoint GLCompute

// SPIRV-DAG: %[[STRUCTNAME:[A-Za-z0-9_]+]]  = OpTypeStruct %int %_Array_natural_vector_int_3_2

// SPIRV-DAG: OpMemberDecorate %[[STRUCTNAME:[A-Za-z0-9_]+]] 1 Offset 4
