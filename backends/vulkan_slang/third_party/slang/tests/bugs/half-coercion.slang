//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK-NOT: warning

RWStructuredBuffer<half> output;
[numthreads(1,1,1)]
void computeMain()
{
    output[0] = 0; // coercion from 0 to half should not result in a warning.
}