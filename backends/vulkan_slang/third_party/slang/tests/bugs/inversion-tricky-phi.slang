//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -target hlsl

// Annoyingly, the reproducer for this doesn't terminate, so just check that we
// succeeded compilation.
// Previously, this would fail in SCCP after loop inversion failed to account
// for condition variables being used in the loop.
// CHECK: computeMain

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint i : SV_GroupIndex)
{
    float x = 0;
    f(x);
    outputBuffer[i] = x;
}

void f(inout float r)
{
    r = 0;
    float a = 0;
    do {
        do {
            a = a + 1;
            if (a > 0)
                break;
        } while (true);
        r = a;
    } while (true);
}
