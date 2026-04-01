// Test that we allow type conformances whose base interface is generic.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-dx11 -compute  -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute  -output-using-type

public interface ITestInterface<Real : IFloat> {
    Real sample();
}

struct TestInterfaceImpl<Real : IFloat> : ITestInterface<Real> {
    Real sample() {
        return x;
    }
    Real x;
}

//TEST_INPUT: set data = new StructuredBuffer<ITestInterface<float> >[new TestInterfaceImpl<float>{1.0}];
StructuredBuffer<ITestInterface<float>> data;

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4);
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT: type_conformance TestInterfaceImpl<float>:ITestInterface<float> = 3

[numthreads(1, 1, 1)]
void computeMain()
{
    let obj = data[0];
    // CHECK: 1
    outputBuffer[0] = int(obj.sample());
}