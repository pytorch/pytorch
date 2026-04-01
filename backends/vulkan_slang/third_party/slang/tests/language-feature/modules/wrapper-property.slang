//TEST:SIMPLE(filecheck=CHECK): -target spirv

//CHECK: OpEntryPoint

public interface ITest {
    public property int val;
};

public struct TestImpl : ITest {
    public int val;
}

public struct Test : ITest = TestImpl;

StructuredBuffer<Test> data;
RWStructuredBuffer<int> output;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain()
{
    int val = data[0].val;
    output[0] = val;
}