//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry computeMain -stage compute

// CHECK: coherent
// CHECK-NOT: coherent

globallycoherent RWByteAddressBuffer bab;
RWStructuredBuffer<uint> output;
void computeMain()
{
    output[0] = bab.Load<uint>(0);
}