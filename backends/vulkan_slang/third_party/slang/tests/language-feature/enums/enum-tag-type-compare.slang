// enum-tag-type-compare.slang

// Test comparisons for `enum`s with explicit tag types

//TEST(compute):COMPARE_COMPUTE:

enum class MaterialType : uint32_t
{
    A,
    B,
    C,
    D,
}

struct Material
{
    MaterialType getType() { return MaterialType(_type); }

    uint _type;
}

uint test(uint val)
{
    Material m = { val };

    MaterialType b = MaterialType.B;

    int result = 0x100000;
    if(m.getType() == b) result += 0x10;
    if(m.getType() != b) result += 0x100;
    if((uint)m.getType() == (uint)b) result += 0x1000;
    if((uint)m.getType() != (uint)b) result += 0x10000;

    return result;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    uint inVal = tid;
    uint outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
