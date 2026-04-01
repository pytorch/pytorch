//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type

// CHECK: type: float
// CHECK-NEXT: 1.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 9.000000
// CHECK-NEXT: 16.000000

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

void squareCoopVecElements(CoopVec<float, 4> vec)
{
    for (int i = 0; i < 4; ++i)
    {
        vec[i] = vec[i] * vec[i];
    }
    vec.store(outputBuffer, 0);
}

[numthreads(1, 1, 1)]
void computeMain()
{
    CoopVec<float, 4> vec = CoopVec<float, 4>(1.0, 2.0, 3.0, 4.0);
    squareCoopVecElements(vec);
}
