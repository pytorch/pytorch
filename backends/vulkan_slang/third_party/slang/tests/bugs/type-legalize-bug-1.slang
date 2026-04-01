//TEST(compute):COMPARE_COMPUTE: -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl

//TEST_INPUT:ubuffer(data=[9 9 9 9], stride=4):out,name outputBuffer
//TEST_INPUT:type_conformance A:IFoo=0
//TEST_INPUT:type_conformance B:IFoo=1

RWStructuredBuffer<int> outputBuffer;
interface IFoo
{
    associatedtype T : IFoo;
    T getT();
    void doSomething();
}

A createA() { return {}; }
B createB() { return {}; }
ParameterBlock<B> gB;
void user()
{
    IFoo a = createDynamicObject<IFoo>(0, 0);
    IFoo b = createDynamicObject<IFoo>(1, 0);
    test(a.getT(), b);
    test(a, gB.getT());
}
B test<T:IFoo>(T a, IFoo b)
{
    a.doSomething();
    b.doSomething();
    return {};
}
struct B :IFoo
{
    A a;
    typealias T = A;
    T getT() { return {};}
    void doSomething()
    {
        outputBuffer[0] = 1;
    }
}
struct A : IFoo
{
    typealias T = B;
    T getT() { return {};}
    void doSomething()
    {
        outputBuffer[0] = 1;
    }
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    user();
}
