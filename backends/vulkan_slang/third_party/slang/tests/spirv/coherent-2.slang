// Test that globallycoherent works on arrays of uavs.

//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main -emit-spirv-directly

globallycoherent RWByteAddressBuffer buffer[];

RWStructuredBuffer<float> output;
[numthreads(4,1,1)]
void main(int tid : SV_DispatchThreadID)
{
    buffer[NonUniformResourceIndex(0)].InterlockedAdd(0, 1);
    AllMemoryBarrier();
    output[tid] = buffer[0].Load(0);
    // CHECK-DAG: OpDecorate %buffer Coherent
    // CHECK-DAG: OpDecorate %[[VAR1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK: %[[VAR1]] = OpAccessChain %{{.*}} %buffer
}
