//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type

// CHECK:      123
// CHECK-NEXT: 4567
// CHECK-NEXT: -10
// CHECK-NEXT: 16772649

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct S {
    int foo : 8;
    uint bar : 24;
};

[numthreads(1, 1, 1)]
void computeMain()
{
    S s;
    s.foo = 123;
    s.bar = 4567;
    outputBuffer[0] = s.foo;
    outputBuffer[1] = s.bar;

    s.foo *= 2;
    s.bar *= 0xffffff;
    outputBuffer[2] = s.foo;
    outputBuffer[3] = s.bar;
}
