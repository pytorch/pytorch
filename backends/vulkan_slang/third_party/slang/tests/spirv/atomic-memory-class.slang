//TEST:SIMPLE(filecheck=CHECK): -target spirv

uniform Atomic<int>* pInt;
RWStructuredBuffer<Atomic<int>> bInt;
static groupshared Atomic<int> sInt;

[numthreads(4,1,1)]
void computeMain()
{
    // CHECK: OpAtomicLoad {{.*}} %uint_258
    // CHECK: OpAtomicStore {{.*}} %uint_68
    // CHECK: OpAtomicStore {{.*}} %uint_260
    let v = sInt.load(MemoryOrder.Acquire);
    pInt->store(v, MemoryOrder.Release);
    sInt.store(v, MemoryOrder.Release);
}