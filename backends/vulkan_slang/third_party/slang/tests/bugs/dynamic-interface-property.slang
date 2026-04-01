//TEST:COMPARE_COMPUTE(filecheck-buffer=CHECK):-output-using-type

public interface ITest {
    property int value;
};

struct Test : ITest {
    int value;
};

//TEST_INPUT: type_conformance Test:ITest = 1;

//TEST_INPUT: set inputBuffer = ubuffer(data=[0 0 1 0 1 0 0 0], stride=4)
RWStructuredBuffer<ITest> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[NumThreads(1, 1, 1)]
void computeMain()
{
    // CHECK: 1
    outputBuffer[0] = inputBuffer[0].value;
}
