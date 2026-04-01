// https://github.com/shader-slang/slang/issues/4476

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cuda -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

namespace A
{
    struct Struct1<let SIZE : uint>
    {
        uint data;
    };

    Struct1<Z1> myFunc<let Z0 : uint, let Z1 : uint>(Struct1<Z0> inputS1)
    {
        Struct1<Z1> s1;
        s1.data = inputS1.data + 2U;
        return s1;
    }
};


A::Struct1<Z1> myFunc<let Z0 : uint, let Z1 : uint>(A::Struct1<Z0> inputS1)
{
    A::Struct1<Z1> s1;
    s1.data = inputS1.data + 5U;
    return s1;
}

namespace A
{
    struct Struct2<let SIZE : uint>
    {
        Struct1<SIZE> s1;
    }

    Struct2<Z1> myFunc<let Z0 : uint, let Z1 : uint>(Struct2<Z0> inputS2)
    {
        Struct2<Z1> s2;
        // We want to cover a corner case in our compiler where:
        // when looking up "myFunc", the compiler should find
        //   Struct1<Z1> A::myFunc<let Z0 : uint, let Z1 : uint>(Struct1<Z0> inputS1)
        // and it won't be ambiguous with the global "myFunc".
        s2.s1 = myFunc<Z0, Z1>(inputS2.s1);
        return s2;
    }
};

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain(uint3 threadID: SV_DispatchThreadID)
{
    using namespace A;

    Struct2<10> input = {{threadID.x}};

    Struct2<20> output;
    output = myFunc<10, 20>(input);
    outputBuffer[0] = output.s1.data;

    // BUF: 2
}
