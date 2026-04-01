//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

// Test that an associated type can be constrained to a generic interface.


interface IB<let DIM : int>
{
    int bar();
}

interface IA<let DIM : int>
{
    associatedtype B : IB<DIM>;
    B getB();
}

struct BImpl<int x> : IB<x>
{
    int bar() { return x; }
}

struct AImpl<int y> : IA<y>
{
    typealias B = BImpl<y>;
    B getB() { BImpl<y> b = {}; return b; }
}

int test<int z, T : IA<z>>(T t)
{
    T.B bb = t.getB();
    return bb.bar();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    AImpl<5> a;
    // CHECK: 5
    outputBuffer[0] = test(a);
}