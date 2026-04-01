//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type

// CHECK: type: float
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 8.000000

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

void processCoopVec(CoopVec<float, 4> vec)
{
    (vec * 2.0).store(outputBuffer, 0);
}

[numthreads(1, 1, 1)]
void computeMain()
{
    let vec = CoopVec<float, 4>(1.0, 2.0, 3.0, 4.0);
    processCoopVec(vec);
}
