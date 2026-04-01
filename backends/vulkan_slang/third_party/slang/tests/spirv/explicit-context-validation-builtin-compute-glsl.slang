//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -target spirv -emit-spirv-directly -stage compute -entry computeMain -enable-experimental-passes

// Check to ensure builtin is not moved into a kernelContext (part of entry-point). Ensure builtin is referenced directly.

// CHECK: OpEntryPoint {{.*}} %gl_NumWorkGroups
// CHECK: OpDecorate %gl_NumWorkGroups BuiltIn NumWorkgroups
// CHECK: %gl_NumWorkGroups = OpVariable {{.*}} Input
// CHECK: %[[NUM_WORK_GROUP_LOAD:[A-Za-z0-9_]+]] = OpLoad %v3uint %gl_NumWorkGroups
// CHECK: OpCompositeExtract %uint %[[NUM_WORK_GROUP_LOAD]] 0

RWStructuredBuffer<uint> outputBuffer;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
void computeMain()
{
    outputBuffer[0] = gl_NumWorkGroups.x;
    outputBuffer[1] = gl_NumWorkGroups.y;
    outputBuffer[2] = gl_NumWorkGroups.z;
}