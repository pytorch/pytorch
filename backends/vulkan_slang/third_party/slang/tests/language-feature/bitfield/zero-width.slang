//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK:      123
// CHECK-NEXT: 4567
// CHECK-NEXT: 1
// CHECK-NEXT: 0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

struct S {
    int foo : 8;
    int breaker : 0;
    int bar : 24;
};

[numthreads(1, 1, 1)]
void computeMain()
{
    S s;
    s.foo = 123;
    s.bar = 4567;
    s.breaker = 9999;
    outputBuffer[0] = s.foo;
    outputBuffer[1] = s.bar;
    outputBuffer[2] = sizeof(S) == sizeof(int) * 2;
    outputBuffer[3] = s.breaker;
}
