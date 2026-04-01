//TEST(compute):COMPARE_COMPUTE:-vk -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name gBuffer
RWStructuredBuffer<int> gBuffer;

int getX(int4 v) { return v.x; }

[numthreads(1, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID, uint GI: SV_GroupIndex)
{
    if (GI< 1)
    {
        uint LdsCoord = GI;
        uint a = gBuffer[LdsCoord];
        LdsCoord += 1;
        gBuffer[LdsCoord] = 1;
    }
}
