//TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry computeMain
RWStructuredBuffer<int> outputBuffer;

//CHECK: error 30851

struct DefaultStructNoInit
{
    int data0 = 2;
    int data1 = 2;

};
extension DefaultStructNoInit
{
}

struct DefaultStructWithInit
{
    int data0;
    int data1 = 3;
    int data2;
};
extension DefaultStructWithInit
{
    __init()
    {
        data0 = 3;
        data2 = 3;
    }
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    DefaultStructNoInit noInit = DefaultStructNoInit();
    DefaultStructWithInit withInit = DefaultStructWithInit();
    // BUF: 1
    outputBuffer[0] = true
        && noInit.data0 == 2
        && noInit.data1 == 2

        && withInit.data0 == 3
        && withInit.data1 == 3
        && withInit.data2 == 3
        ;
}