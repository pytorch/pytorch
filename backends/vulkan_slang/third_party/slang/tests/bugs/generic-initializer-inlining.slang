//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu

// CHECK:      1
// CHECK-NEXT: 0
// CHECK-NEXT: 0
// CHECK-NEXT: 0

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

struct C<T>
{
    [ForceInline]
    __init<U>(U value)
    {
        for (int i = 0; i < 1; ++i)
        {}
    }
}

static C<int> c = C<int>(1);

int use<T>(T x)
{
    return 1;
}

[numthreads(1, 1, 1)]
void computeMain(uint i : SV_GroupIndex)
{
    outputBuffer[0] = use(c);
}

