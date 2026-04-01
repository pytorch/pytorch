//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
[numthreads(1, 1, 1)]
void computeMain(uint tig : SV_GroupIndex)
{
    // Test that we infer the type parameters to X from the type of f
    x(f);
    // Testthat we infer the type paramters to from a specialized generic (g)
    y(g<int>);
    outputBuffer[tig] = p;
}

static int p = 0;

func x<A, B>(f : functype (A) -> B)
{
    ++p;
}

float f(int)
{
    return 1;
}

func y<A, B, C>(g : functype (A, B) -> C)
{
    p += 2;
}

void g<A>(A, bool)
{ }

