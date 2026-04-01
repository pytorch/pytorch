//TEST:SIMPLE(filecheck=CHECK): -stage compute -target spirv -emit-spirv-directly -entry main

RWStructuredBuffer<float> output;
static const int vmin = min(1,2);

// CHECK: OpEntryPoint

[numthreads(1,1,1)]
void main()
{
    output[0] = vmin;
}
