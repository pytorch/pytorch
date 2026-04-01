//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry computeMain
//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry computeMain

// CHECK-NOT: {{.* }}= 0;
// CHECK-NOT: return 0;

RWStructuredBuffer<int> outputBuffer;

[noinline]
int returnInt()
{
    int myInt;
    return myInt;
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
// BUF: 1
    outputBuffer[0] = true
        && returnInt() == 0
        ;
}
