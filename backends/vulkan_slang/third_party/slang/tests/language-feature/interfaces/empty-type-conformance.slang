//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-dx11 -compute  -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute  -output-using-type

interface TestInterface {
    float sample();
}
struct TestImplementation : TestInterface {
    float sample() {
        return 1.0f;
    }
};

//TEST_INPUT: set inBuffer = new StructuredBuffer<TestInterface>[new TestImplementation{}];
StructuredBuffer<TestInterface> inBuffer;

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4);
RWStructuredBuffer<float> outputBuffer;

//TEST_INPUT: type_conformance TestImplementation:TestInterface=0

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    // CHECK: 1.0
    outputBuffer[0] = inBuffer[0].sample();
}