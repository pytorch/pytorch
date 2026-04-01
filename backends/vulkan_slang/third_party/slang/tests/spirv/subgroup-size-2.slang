// Test that using workgroup size from more than one entrypoint result in
// correct lowering into global variable.

//TEST:SIMPLE(filecheck=CHECK_EXPERIMENTAL): -target spirv -emit-spirv-directly -fvk-use-entrypoint-name -O0 -enable-experimental-passes
//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -fvk-use-entrypoint-name -O0



RWStructuredBuffer<int> outputBuffer;

uint3 f() { return WorkgroupSize(); }

[shader("compute")]
[numthreads(1u, 2u, 3)]
void compute1()
{
    // CHECK_EXPERIMENTAL-DAG: %[[VAR:[A-Za-z0-9_]+]] = OpTypePointer Function %v3int
    // CHECK_EXPERIMENTAL: OpAccessChain %[[VAR]]
    //
    // CHECK_EXPERIMENTAL-DAG: %[[CALL_RS:[A-Za-z0-9_]+]] = OpFunctionCall %v3uint %f
    // CHECK_EXPERIMENTAL: OpCompositeExtract %uint %[[CALL_RS]] 0
    //
    // CHECK_EXPERIMENTAL-DAG: %[[PTR:[A-Za-z0-9_]+]] = OpAccessChain %_ptr_StorageBuffer_int %outputBuffer %int_0 %int_1
    // CHECK_EXPERIMENTAL: OpStore %[[PTR]] %int_2

    // CHECK-DAG: %[[VAR:[A-Za-z0-9_]+]] = OpVariable %_ptr_Private_v3int Private
    // CHECK: OpStore %[[VAR]]
    //
    // CHECK-DAG: %[[CALL_RS:[A-Za-z0-9_]+]] = OpFunctionCall %v3uint %f
    // CHECK: OpCompositeExtract %uint %[[CALL_RS]] 0
    //
    // CHECK-DAG: %[[PTR:[A-Za-z0-9_]+]] = OpAccessChain %_ptr_StorageBuffer_int %outputBuffer %int_0 %int_1
    // CHECK: OpStore %[[PTR]] %int_2

    const int x = f().x;
    outputBuffer[0] = x;
    outputBuffer[1] = WorkgroupSize().y;
}

[shader("compute")]
[numthreads(4, 5, 6)]
void compute2()
{
    const int x = f().x;
    outputBuffer[0] = x;
}