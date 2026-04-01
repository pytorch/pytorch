//TEST:SIMPLE(filecheck=CHECK): -target spirv

public interface ITest {
    public void testIn(int a);
    public void testOut(out int b);
};

public struct TestImpl : ITest {
    // CHECK: ([[# @LINE + 1]]): error 38105
    public void testIn(out int a) {
        a = 5;
    }
    // CHECK: ([[# @LINE + 1]]): error 38105
    public void testOut(int b) {
        b = 6;
    }
}

RWStructuredBuffer<int> output;

void doSomething<T>(T data) where T : ITest {
    int a = 516;
    data.testIn(a);
    int b = 687;
    data.testOut(b);

    output[0] = a;
    output[1] = b;
}

[shader("compute")]
[numthreads(1,1,1)]
void computeMain()
{
    TestImpl data;
    doSomething(data);
}