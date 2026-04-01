//TEST:SIMPLE(filecheck=CHECK):-target spirv

// CHECK: OpCapability Int64Atomics

groupshared Atomic<uint64_t> atomicVariable;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint64_t value = 4ll;
    atomicVariable.store(value);
}