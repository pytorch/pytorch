// multiple-namespace.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Multiple namespace open/closing

namespace A
{
    struct Struct1
    {
        int a;
    }
}

namespace A
{
    struct Struct2
    {
        int b;
    }
}

namespace A
{
    struct Struct3
    {
        int c;
    }
}
   
// A few more 

namespace A {}
namespace A {}
namespace A { namespace A {} }
   
int test(int val)
{
    A.Struct1 s1 = {};
    A::Struct2 s2 = { val };
    A.Struct3 s3 = { val * val };
    
    return s1.a + s2.b + s3.c;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
