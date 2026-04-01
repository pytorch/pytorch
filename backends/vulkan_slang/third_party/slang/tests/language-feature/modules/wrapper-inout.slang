//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK): -output-using-type

public interface ITest {
    public int testDir(inout int a);
};

public struct TestImpl : ITest {
    public int testDir(inout int a) {
        int oldA = a;
        a = 5;
        return a;
 }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=output
RWStructuredBuffer<int> output;

public struct Test : ITest = TestImpl;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain()
{
    Test data;
    int a = 516;
    int b = data.testDir(a);
    // CHECK: 5
    output[0] = b;
}