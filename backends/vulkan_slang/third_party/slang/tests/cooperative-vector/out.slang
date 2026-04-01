//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type

// CHECK: type: float
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 8.000000

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

void doubleCoopVec(CoopVec<float, 4> vec, out CoopVec<float, 4> result)
{
    result = vec * 2.0;
}

[numthreads(1, 1, 1)]
void computeMain()
{
    let vec = CoopVec<float, 4>(1.0, 2.0, 3.0, 4.0);
    CoopVec<float, 4> result;
    doubleCoopVec(vec, result);
    result.store(outputBuffer, 0);
}
