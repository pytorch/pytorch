//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK:      123
// CHECK-NEXT: 4567
// CHECK-NEXT: 0
// CHECK-NEXT: 0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

struct S<T> {
    int foo : 8;
    uint bar : 24;
    T baz;
};

[numthreads(1, 1, 1)]
void computeMain()
{
    S<int> s;
    s.foo = 123;
    s.bar = 4567;
    outputBuffer[0] = s.foo;
    outputBuffer[1] = s.bar;
    s.foo = 0;
    s.bar = 0;
    outputBuffer[2] = s.foo;
    outputBuffer[3] = s.bar;
}
