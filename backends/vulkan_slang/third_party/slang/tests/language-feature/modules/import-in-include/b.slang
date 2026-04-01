implementing c;

int f()
{
    return helperFunc();
}

RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    outputBuffer[0] = f();
}
