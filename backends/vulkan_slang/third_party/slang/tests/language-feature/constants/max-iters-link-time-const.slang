//TEST:SIMPLE(filecheck=CHECK): -target spirv
// CHECK: OpEntryPoint

extern static const int num = 10;
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    [MaxIters(num)]
    for (int i = 0; i < num; i++)
    {
        outputBuffer[i] = i;
    }
}
