//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main -allow-glsl
#version 450

[vk::constant_id(1)]
const int constValue1 = 0;

[vk::constant_id(2)]
const int constValue3 = 5;

// CHECK-DAG: OpExecutionModeId %main LocalSizeId %[[C0:[0-9A-Za-z_]+]] %[[C1:[0-9A-Za-z_]+]] %[[C2:[0-9A-Za-z_]+]]
// CHECK-DAG: OpDecorate %[[C0]] SpecId 1
// CHECK-DAG: OpDecorate %[[C1]] SpecId 0
// CHECK-DAG: OpDecorate %[[C2]] SpecId 2

layout(local_size_x_id = 1, local_size_y_id = 0, local_size_z = constValue3) in;
void main()
{
}

