// Test that we can synthesize requirements for generic methods.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-dx11 -compute  -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute  -output-using-type

interface IBase
{
    static float get();
}
interface IBar : IBase
{
    float derivedMethod();
}

struct Bar : IBar
{
    static float get() { return 1.0f; }
    float derivedMethod() { return 2.0f; }
}

interface ITestInterface<Real : IFloat>
{
    Real sample<T : IBar>(T t);

    __init<T : IBar>(T t);

    __generic<T : IBar> 
    __subscript(T t)->Real { get; }
}

struct TestInterfaceImpl<Real : IFloat> : ITestInterface<Real>
{
    // The signature of this sample method is different from the one in the 
    // interface. However, we should be able to form a call into this method
    // from the synthesized implementation matching the interface definition,
    // so the conformance should hold.
    Real sample<T : IBase>(T t)
    {
        return x + Real(T.get());
    }
    
    // Test the same thing for constructors.
    __init<T : IBase>(T t)
    {
        x = Real(T.get());
    }

    // Test the same thing for subscript operators.
    __generic<T : IBase>
    __subscript(T t)->Real { get { return x + Real(T.get()); } }
    Real x;
}

float test(ITestInterface<float> obj)
{
    Bar b = {};
    return obj.sample<Bar>(b);
}

float test1(ITestInterface<float> obj)
{
    Bar b = {};
    return obj[b];
}

float test2<T:ITestInterface<float>>()
{
    Bar b = {};
    T obj = T(b);
    return obj[b];
}

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4);
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    TestInterfaceImpl<float> obj;
    obj.x = 1.0f;

    // CHECK: 2
    outputBuffer[0] = int(test(obj));

    // CHECK: 2
    outputBuffer[1] = int(test1(obj));

    // CHECK: 2
    outputBuffer[3] = int(test2<TestInterfaceImpl<float>>());
}