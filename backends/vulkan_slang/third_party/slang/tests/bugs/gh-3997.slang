//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -O0 -g

//CHECK: OpEntryPoint

float atomicAdd(__ref float value, float amount)
{
    __target_switch
    {
    case cpp:
        __requirePrelude("#include <atomic>");
        __intrinsic_asm "std::atomic_ref(*$0).fetch_add($1)";
    case spirv:
        return __atomic_add(value, amount);
    }
}

RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
[shader("compute")]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID) {
    atomicAdd(outputBuffer[0], 1);
}