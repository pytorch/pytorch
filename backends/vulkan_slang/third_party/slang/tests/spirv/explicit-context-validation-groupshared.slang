//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -stage compute -entry computeMain -enable-experimental-passes

// Check to ensure builtin is not moved into a kernelContext (part of entry-point). Ensure builtin is referenced directly.

// CHECK: OpEntryPoint{{.*}} %val
// CHECK: %[[GROUP_SHARED_VAR:(.*)]] = OpVariable{{.*}} Workgroup

// CHECK: OpControlBarrier
// CHECK: OpControlBarrier
// CHECK-DAG: %[[GROUP_SHARED_VAL:(.*)]] = OpLoad{{.*}} %[[GROUP_SHARED_VAR]]
// CHECK-DAG: %[[OUTPUT_BUFFER_LOC:(.*)]] = OpAccessChain{{.*}} %outputBuffer
// CHECK: OpStore %[[OUTPUT_BUFFER_LOC]] %[[GROUP_SHARED_VAL]]

groupshared uint val;
RWStructuredBuffer<uint> outputBuffer;

void nestedCall(uint index)
{
    val += 1;
    GroupMemoryBarrierWithGroupSync();
    outputBuffer[index] = val;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    val = 0;
    GroupMemoryBarrierWithGroupSync();
    nestedCall(dispatchThreadID.x);
}