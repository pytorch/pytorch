//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cuda -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;


interface IFoo
{
    int foo(int a);
}

struct MyImpl1 : IFoo
{
    int foo(int a) { return a; }
}

struct MyImpl2 : IFoo
{
    int foo(int a) { return a + 5; }
}

int test(IFoo foo, int idx)
{
    int val = 0;
    if (let a = foo as MyImpl1)
    {
        val = a.foo(idx);
    }
    else if (let a = foo as MyImpl2)
    {
        val = a.foo(idx);
    }
    return (val);
}

int test1<T>(T t)
{
    if (let a = t as uint)
    {
        return 1;
    }
    else if(let a = t as float)
    {
        return 2;
    }
    else if (let a = t as double)
    {
        return 3;
    }
    else if (let a = t as int)
    {
        return 4;
    }
    else if (let a = t as uint64_t)
    {
        return 5;
    }
    else
    {
        return 6;
    }
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    MyImpl1 impl1;
    MyImpl2 impl2;
    // CHECK: 1
    // CHECK: 7
    outputBuffer[0] = test(impl1, 1);
    outputBuffer[1] = test(impl2, 2);

    // CHECK: 1
    outputBuffer[2] = test1(2U);
    // CHECK: 2
    outputBuffer[3] = test1(2.0f);
    // CHECK: 3
    outputBuffer[4] = test1(2.0lf);
    // CHECK: 4
    outputBuffer[5] = test1(2);
    // CHECK: 5
    outputBuffer[6] = test1(2LLU);
    // CHECK: 6
    outputBuffer[7] = test1(impl1);
}
