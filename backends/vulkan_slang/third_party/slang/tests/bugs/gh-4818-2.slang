//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry computeMain -stage compute

// CHECK: coherent
// CHECK-NOT: coherent

globallycoherent StructuredBuffer<uint> sb;
RWStructuredBuffer<uint> output;
void computeMain()
{
    output[0] = sb.Load(0);
}