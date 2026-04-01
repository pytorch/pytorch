// bit-field.slang

//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-vk -compute -output-using-type

// Test direct SPIR-V emit on bit field operations.

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

enum Flag
{
    AAA = 1,
    BBB = 2,
    CCC = 4,
}
struct Data
{
    uint flags;
    [mutating] void setFlag(Flag g, bool v)
    {
        const uint bit = uint(g) << 8;
        if (v) flags |= bit;
        else flags &= ~bit;
    }
    bool hasFlag(Flag g)
    {
        const uint bit = uint(g) << 8;
        return (flags & bit) != 0;
    }
}

//TEST_INPUT: set Uniforms.d = new Data{0};
cbuffer Uniforms
{
    Data d;
}

// BUFFER: 1
// BUFFER: 1
// BUFFER: 1
// BUFFER: 0

[numthreads(1,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Data d1 = d;
    d1.setFlag(Flag.BBB, true);
    d1.setFlag(Flag.AAA, true);
    resultBuffer[0] = d1.hasFlag(Flag.AAA) ? 1 : 0;
    resultBuffer[1] = d1.hasFlag(Flag.BBB) ? 1 : 0;
    d1.setFlag(Flag.BBB, false);
    resultBuffer[2] = d1.hasFlag(Flag.AAA) ? 1 : 0;
    resultBuffer[3] = d1.hasFlag(Flag.BBB) ? 1 : 0;
}
