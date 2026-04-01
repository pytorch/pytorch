//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test here is to try and associate a value with a type

This doesn't work without 'public' on struct B. This is necessary such that
those implementations of `IGetE` are visible for dynamic dispatch.

Otherwise the error 

> tests/experiments/generic/type-to-value-4.slang(21): error 50100: No type conformances are found for interface 'IGetE'. Code generation for current target requires at least one implementation type present in the linkage.
> interface IGetE
          ^~~~~

This is somewhat 'surprising' because the code *explicitly* instanciates B, so why is it necessary to make it 
public?

If we make both struct A and struct B public, the following code uses dynamic dispatch in order to implement e.getE(). When it does this it introduces

```HLSL
struct Tuple_0
{
    uint2 value0_0;
    uint2 value1_0;
    AnyValue16 value2_0;
};

Tuple_0 _S1;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID_0 : SV_DISPATCHTHREADID)
{
    int _S7 = U_S3tu05IGetE4getEp0p3tu04Enum_0(_S1.value1_0);

    // ...
}
```

But _S1 is never intialized (in HLSL it becomes a Global in a constant buffer). Why does creating a local variable produce something in a constant buffer?

*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum class Enum
{
    A, B
};

interface IGetE
{
	static Enum getE();    
};

struct A : IGetE
{
    static Enum getE() { return Enum::A; }
};

struct B : IGetE
{
    static Enum getE() { return Enum::B; }
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
   
    // Err.. even though IGetE doesn't require an instanciation, not clear how to set it. So lets try with instanciation
    B b;
    IGetE g = b;

    let e = g.getE();

    outputBuffer[dispatchThreadID.x] = int(e);
}
