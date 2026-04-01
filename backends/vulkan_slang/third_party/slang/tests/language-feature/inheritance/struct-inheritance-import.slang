// struct-inheritance-import.slang

//TEST(compute):COMPARE_COMPUTE:

// This test confirms that the synthesized code for struct type inheritance
// includes proper linkage decorations so that it can work across modules.

import struct_inheritance_imported;

int test(int val)
{
    Derived d;
    d.a = val;

    return 0x100 + d.a*0x10 + getA(d);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    int inVal = int(tid);
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
