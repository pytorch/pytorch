// glsl-int-swizzle.slang

//TEST(compute):COMPARE_COMPUTE:-vk -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name gBuffer
RWStructuredBuffer<int> gBuffer;

int getX(int4 v) { return v.x; }

[numthreads(4,1,1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    gBuffer[tid.x] = getX((1).xxxx);
}
