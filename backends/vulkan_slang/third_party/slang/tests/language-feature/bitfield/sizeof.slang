//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-mtl -output-using-type


// CHECK:      1
// CHECK-NEXT: 1
// CHECK-NEXT: 1
// CHECK-NEXT: 1

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

struct S {
    int foo : 8;
    int bar : 24;
};

// This packs the smaller int into the int64
struct T {
    int64_t foo : 33;
    int bar : 24;
};

// This takes two ints to store all the bits
struct P {
    int foo : 24;
    int bar : 24;
};

// Even though the smaller field comes first, it's still packed with the larger
// one
struct Q {
    int8_t foo : 1;
    int64_t bar : 63;
};

[numthreads(1, 1, 1)]
void computeMain()
{
    outputBuffer[0] = sizeof(S) == sizeof(int);
    outputBuffer[1] = sizeof(T) == sizeof(int64_t);
    outputBuffer[2] = sizeof(P) == sizeof(int) * 2;
    outputBuffer[3] = sizeof(Q) == sizeof(int64_t);
}
