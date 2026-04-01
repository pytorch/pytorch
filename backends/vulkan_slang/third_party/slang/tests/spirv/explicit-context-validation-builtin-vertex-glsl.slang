//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -target spirv -emit-spirv-directly -stage vertex -entry vertexMain -enable-experimental-passes

// Check to ensure varying output/input and builtin is not moved into a kernelContext (part of entry-point). Ensure varying output/input and builtin is referenced directly.

// CHECK: OpEntryPoint{{((.*)(%.*data1In|%.*data2In|%.*data1Out)(.*))|((.*)(%.*data1In|%.*data2In|%.*data1Out)(.*))|((.*)(%.*data1In|%.*data2In|%.*data1Out)(.*))}}

//CHECK-DAG: OpDecorate %[[DATA_IN_1:.*data1In]] Location 1
//CHECK-DAG: OpDecorate %[[DATA_IN_2:.*data2In]] Location 2
//CHECK-DAG: OpDecorate %[[DATA_OUT_1:.*data1Out]] Location 0
//CHECK-DAG: OpDecorate %gl_Position BuiltIn Position

//CHECK-DAG: %[[DATA_IN_1]] = OpVariable{{.*}} Input
//CHECK-DAG: %[[DATA_IN_2]] = OpVariable{{.*}} Input
//CHECK-DAG: %[[DATA_OUT_1]] = OpVariable{{.*}} Output

//CHECK-DAG: OpLoad{{.*}} %[[DATA_IN_1]]
//CHECK-DAG: OpLoad{{.*}} %[[DATA_IN_2]]

//CHECK-DAG: OpStore %gl_Position
//CHECK-DAG: OpStore %[[DATA_OUT_1]]

layout(location = 1) in vec4 data1In;
layout(location = 2) in int data2In;
layout(location = 0) out int data1Out;

void nestedCall()
{
    gl_Position = data1In;
    data1Out = data2In;
}

void vertexMain()
{
    nestedCall();
}
