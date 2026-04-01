//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -emit-spirv-via-glsl
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -profile cs_6_7 -dx12 -use-dxil -shaderobj -render-feature hardware-device
//TEST(compute):COMPARE_COMPUTE_EX:-metal -compute -shaderobj -xslang -DMETAL

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(16, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = WaveGetLaneIndex();

    if (index < 4)
    {
        // Quad 1.
        // Should return true, index 0's expr is true while all other indices' expr are false.
        outputBuffer[index] = uint(QuadAny((index % 4) == 0));
    }
    else if (index < 8)
    {
        // Quad 2.
        // Should return false, all indices' expr are false.
        bool falseCondition = (5 == 4);
        outputBuffer[index] = uint(QuadAny(falseCondition));
    }
    else if (index < 12)
    {
        // Quad 3.
        // Should return false, index 0's expr is true while all other indices' expr are false.
        outputBuffer[index] = uint(QuadAll((index % 4) == 0));
    }
    else
    {
        // Quad 4.
        // Should return true, all indices' expr are true.
        bool trueCondition = (5 == 5);
        outputBuffer[index] = uint(QuadAll(trueCondition));
    }
}

