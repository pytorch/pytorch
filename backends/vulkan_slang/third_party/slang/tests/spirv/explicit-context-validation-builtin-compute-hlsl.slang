//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -stage compute -entry computeMain -enable-experimental-passes

// Check to ensure builtin is not moved into a kernelContext (part of entry-point). Ensure builtin is referenced directly.

// CHECK: OpEntryPoint {{.*}} %gl_GlobalInvocationID
// CHECK: OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
// CHECK: %gl_GlobalInvocationID = OpVariable {{.*}} Input
// CHECK: %[[NUM_WORK_GROUP_LOAD:[A-Za-z0-9_]+]] = OpLoad %v3uint %gl_GlobalInvocationID
// CHECK: OpCompositeExtract %uint %[[NUM_WORK_GROUP_LOAD]] 0

RWStructuredBuffer<uint> outputBuffer;
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[dispatchThreadID.x] = 1;
}