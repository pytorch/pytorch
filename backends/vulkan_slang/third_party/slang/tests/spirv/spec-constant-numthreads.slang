//TEST:SIMPLE(filecheck=GLSL): -target glsl -allow-glsl
//TEST:SIMPLE(filecheck=GLSL): -target glsl
//TEST:SIMPLE(filecheck=CHECK): -target spirv -allow-glsl
//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK-DAG: OpExecutionModeId %computeMain1 LocalSizeId %[[C0:[0-9A-Za-z_]+]] %[[C1:[0-9A-Za-z_]+]] %[[C2:[0-9A-Za-z_]+]]
// CHECK-DAG: OpDecorate %[[C0]] SpecId 1
// CHECK-DAG: OpDecorate %[[C1]] SpecId 0
// CHECK-DAG: %[[C2]] = OpConstant %int 4
// CHECK-DAG: OpStore %{{.*}} %[[C0]]
// CHECK-DAG: OpStore %{{.*}} %[[C1]]
// CHECK-DAG: OpStore %{{.*}} %[[C2]]

// GLSL-DAG: layout(constant_id = 1)
// GLSL-DAG: int constValue0_0 = 0;
// GLSL-DAG: layout(constant_id = 0)
// GLSL-DAG: int constValue1_0 = 0;
// GLSL-DAG: layout(local_size_x_id = 1, local_size_y_id = 0, local_size_z = 4) in;

[vk::specialization_constant]
const int constValue0 = 0;

[vk::constant_id(0)]
const int constValue1 = 0;

RWStructuredBuffer<int> outputBuffer;

[numthreads(constValue0, constValue1, 4)]
void computeMain1()
{
    int3 size = WorkgroupSize();
    outputBuffer[0] = size.x;
    outputBuffer[1] = size.y;
    outputBuffer[2] = size.z;
}
