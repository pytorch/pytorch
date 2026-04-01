//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -render-feature cooperative-vector -output-using-type

// CHECK: type: float
// CHECK-NEXT: 1.000000
// CHECK-NEXT: 2.000000
// CHECK-NEXT: 3.000000
// CHECK-NEXT: 4.000000
// CHECK-NEXT: 5.000000
// CHECK-NEXT: 6.000000
// CHECK-NEXT: 7.000000
// CHECK-NEXT: 8.000000

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct MyStruct
{
    CoopVec<float, 4> vec1;
    CoopVec<float, 4> vec2;
};

[numthreads(1, 1, 1)]
void computeMain()
{
    MyStruct s;
    s.vec1 = CoopVec<float, 4>(1.0, 2.0, 3.0, 4.0);
    s.vec2 = CoopVec<float, 4>(5.0, 6.0, 7.0, 8.0);

    s.vec1.store(outputBuffer, 0);
    s.vec2.store(outputBuffer, 16);
}
