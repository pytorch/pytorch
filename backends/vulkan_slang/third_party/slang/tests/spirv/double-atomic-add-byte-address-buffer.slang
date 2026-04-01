//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpAtomicFAddEXT

RWByteAddressBuffer bab;

[numthreads(1, 1, 1)]
void computeMain()
{
    double d;
    bab.InterlockedAddF64(0, 1.0, d);
}