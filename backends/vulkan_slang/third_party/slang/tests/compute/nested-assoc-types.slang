// Test calling differentiable function through dynamic dispatch.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[anyValueSize(16)]
interface IFoo
{
    float foo();
}

[anyValueSize(16)]
interface INestedInterface
{
    associatedtype NestedAssocType : IFoo;
}

[anyValueSize(16)]
interface IInterface
{
    associatedtype MyAssocType : INestedInterface;
    MyAssocType.NestedAssocType calc(float x);
}

// ================================

struct A_Assoc_Assoc : IFoo
{
    float a;

    float foo()
    {
        return a;
    }
}

struct A_Assoc : INestedInterface
{
    typedef A_Assoc_Assoc NestedAssocType;
}

struct A : IInterface
{
    typedef A_Assoc MyAssocType

    int data1;

    __init(int data1) { this.data1 = data1; }

    A_Assoc_Assoc calc(float x) { return { x * x * x * data1 }; }
};

// ================================

struct B_Assoc_Assoc : IFoo
{
    float b;

    float foo()
    {
        return b;
    }
}

struct B_Assoc : INestedInterface
{
    typedef B_Assoc_Assoc NestedAssocType;
}

struct B : IInterface
{
    typedef B_Assoc MyAssocType

    int data1;
    int data2;

    __init(int data1, int data2) { this.data1 = data1; this.data2 = data2; }

    B_Assoc_Assoc calc(float x) { return { x * x * data1 * data2 }; }
};

// ================================

float doThing(IInterface obj, float x)
{
    let o = obj.calc(x);
    return o.foo();
}

float f(uint id, float x)
{
    IInterface obj;

    switch (id)
    {
        case 0:
            obj = A(2);
            break;

        default:
            obj = B(2, 3);
    }

    return doThing(obj, x);
}

//TEST_INPUT: type_conformance A:IInterface = 0
//TEST_INPUT: type_conformance B:IInterface = 1

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = f(dispatchThreadID.x, 1.0); // A.calc, expect 2
    outputBuffer[1] = f(dispatchThreadID.x + 1, 1.5); // B.calc, expect 13.5
}