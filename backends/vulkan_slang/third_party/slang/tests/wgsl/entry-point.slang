//TEST:SIMPLE(filecheck=CHECK): -target wgsl-spirv-asm
//CHECK: OpEntryPoint

RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    outputBuffer[0] = 42;
}
