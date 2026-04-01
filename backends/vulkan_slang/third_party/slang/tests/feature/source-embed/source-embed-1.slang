//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -profile cs_5_1 -line-directive-mode none -source-embed-style default -emit-spirv-via-glsl

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK: const uint32_t data[] =
// CHECK: 0x07230203, 0x00010000,
// CHECK: const size_t data_sizeInBytes =

int calcThing(int offset)
{
    int total = 0;
    int another[2] = { 1, 2};
    
    for (int k = 0; k < 20; ++k)
    {       
        for (int i = 0; i < 17; ++i)
        {
            another[i & 1] += k + i;
        }
        
        total += another[k & 1];
        
        if ((k + 7) % 5 == 4)
        {
            return 1;
        }
    }
    
    if (total > 4)
    {
        total = -total;
    }
    
    return total;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = int(dispatchThreadID.x);

    outputBuffer[index] = calcThing(index);
}
