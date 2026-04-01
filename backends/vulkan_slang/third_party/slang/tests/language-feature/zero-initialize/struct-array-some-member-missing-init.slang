//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry computeMain -xslang -zero-initialize
//DISABLE_TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry computeMain -xslang -zero-initialize

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

Array<MyStruct,2> getStructs()
{
    Array<MyStruct, 2> outData;
    return outData;
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    Array<MyStruct,2> myStruct = getStructs();

// BUF: 1
    outputBuffer[0] = true
        && myStruct[0].a == 1
        && myStruct[0].b == 0
        && myStruct[0].c == 1
        && myStruct[0].d == 0

        && myStruct[1].a == 1
        && myStruct[1].b == 0
        && myStruct[1].c == 1
        && myStruct[1].d == 0
        ;
}
