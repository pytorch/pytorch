//DISABLE_TEST:REFLECTION: -profile cs_5_0 -entry computeMain -target cpp 
//TEST(compute):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-d3d12 -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-d3d11 -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj

struct Thing
{
    int a;
    float b;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:ubuffer(data=[7 2 9 53], stride=4):name buffer0
RWStructuredBuffer<int> buffer0;

//TEST_INPUT:ubuffer(data=[23 2], stride=4):name buffer1
RWStructuredBuffer<int> buffer1;

//TEST_INPUT:ubuffer(data=[-10 17 9 4 2 0], stride=4):name buffer2
RWStructuredBuffer<float> buffer2;

//TEST_INPUT:ubuffer(data=[1 0 3 0 7 0], stride=8):name buffer3
RWStructuredBuffer<Thing> buffer3;

//TEST_INPUT:ubuffer(data=[1 0 3 0 7 6], stride=0):name buffer4
RWByteAddressBuffer buffer4;

[numthreads(5, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = int(dispatchThreadID.x);
    int last = -1;
    
    uint count = 0, stride = 0;
    if (index == 0)
    {
        buffer0.GetDimensions(count, stride);
        last = buffer0.Load(int(count - 1));
    }
    else if (index == 1)
    {
        buffer1.GetDimensions(count, stride);
        last = buffer1[count - 1];
    }
    else if (index == 2)
    {
        buffer2.GetDimensions(count, stride);
        last = int(buffer2.Load(int(count - 1)));
    }
    else if (index == 3)
    {
        buffer3.GetDimensions(count, stride);
        
        last = buffer3.Load(int(count - 1)).a;
    }
    else
    {
        buffer4.GetDimensions(count);
        last = buffer4.Load(count - 4);
    }
	outputBuffer[index * 2] = int(count);
    outputBuffer[index * 2 + 1] = last;
}