//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

// CHECK: error 55201:

float test<let x : int>()
{
    return test<x>();
}

RWStructuredBuffer<float> outputBuffer;
[numthreads(1,1,1)]
void main()
{
    outputBuffer[0] = test<1>();
}