// struct-bit-cast-2.slang
// Test bit casts for read across boundaries of scalar types.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -cuda -shaderobj

// Test that bit_cast works for bit-reinterpreting one struct type as another.

struct Foo
{
    uint16_t a;
    uint16_t b;
    uint32_t c;
}

struct Bar
{
    uint64_t d;
}


//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    Bar b = { 0x1122334455667788 };
    Foo f = reinterpret<Foo>(b);
    // CHECK: 7788
    outputBuffer[0] = f.a;
    // CHECK: 5566
    outputBuffer[1] = f.b;
    // CHECK: 11223344
    outputBuffer[2] = f.c;

    uint8_t4 v = {0x10, 0x20, 0x30, 0x40};
    uint32_t u = reinterpret<uint32_t>(v);

    // CHECK: 40302010
    outputBuffer[3] = u;
    int* ptr = reinterpret<int*>(b.d);
    Foo f1 = reinterpret<Foo>(ptr);
    // CHECK: 11223344
    outputBuffer[4] = f1.c;
}
