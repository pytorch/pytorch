//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

interface IGetter
{
    int get();
}
struct Gen0 : IGetter
{
    int get() { return 1; }
};
struct Gen1<TGetter : IGetter> : IGetter
{
    TGetter g = TGetter();
    int get() { return g.get(); }
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

void writeArray(inout float3 a[4])
{
    a[0] = float3(1, 1, 1);
    a[1] = float3(1, 1, 1);
    a[2] = float3(1, 1, 1);
    a[3] = float3(1, 1, 1);    
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Gen1<Gen1<Gen1<Gen0>>> g;
    Gen1<Gen1<Gen1<Gen1<Gen0>>>> g2;
    Gen1<Gen1<Gen0>> g3;

    int a = 0; 
    int b = 5;
    if (a< b && b > a)
	    outputBuffer[dispatchThreadID.x] = (g.get() >> 1) + g2.get() + g3.get();
}
