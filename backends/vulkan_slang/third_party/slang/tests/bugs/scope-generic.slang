//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

struct Test1
{
    struct A<let N : uint = 2>
    {
        static uint num() { return N; }
    };

    static uint num() { return A<2>::num(); }

    typealias B = A<2>;
};

struct Test2
{};

extension Test2
{
    struct A<let N : uint = 2>
    {
        static uint num() { return N; }
    };

    static uint num()
    {
        return A<2>::num(); 
    }

    typealias B = A<2>;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer[0] = Test1::num();
    outputBuffer[1] = Test1::B::num();
    outputBuffer[2] = Test1::A<2>::num(); 

    outputBuffer[3] = Test2::num(); 
    outputBuffer[4] = Test2::B::num();
    outputBuffer[5] = Test2::A<2>::num(); 
}