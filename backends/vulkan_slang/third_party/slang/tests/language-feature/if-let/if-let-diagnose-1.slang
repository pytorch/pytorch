//TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=CHECK): -target cuda -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=CHECK): -target cpp -stage compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;


interface IFoo
{
    int foo(int a);
}

struct MyImpl : IFoo
{
    int foo(int a) { return a; }
}

struct MyImpl1 : IFoo
{
    int foo(int a) { return a; }
}

int test(IFoo foo, int idx)
{
    int val = 0;
    if (let a = foo as MyImpl)
    {
        val = a.foo(idx);
    }
    // CHECK: error 30015: undefined identifier 'a'.
    else if(a == none)
    {
        val = -1;
    }
    else
    {
        // CHECK: error 30015: undefined identifier 'a'.
        if (a == none)
        {
            val = -1;
        }
    }
    return (val);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    MyImpl1 impl;
    outputBuffer[dispatchThreadID.x] = test(impl, dispatchThreadID.x);
}
