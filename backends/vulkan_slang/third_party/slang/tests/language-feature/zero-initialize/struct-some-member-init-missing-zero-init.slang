//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry computeMain
//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry computeMain

// CHECK-NOT: {{.* }}= 0;

RWStructuredBuffer<int> outputBuffer;

struct MyStruct_base
{
    int a = 1;
    int b;
};
struct MyStruct : MyStruct_base
{
    int c = 1;
    int d;
};
MyStruct getStruct()
{
    MyStruct myStruct;
    return myStruct;
}
[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    MyStruct myStruct = getStruct();

// BUF: 1
    outputBuffer[0] = true
        && myStruct.a == 1
        && myStruct.b == 0
        && myStruct.c == 1
        && myStruct.d == 0
        ;
}
