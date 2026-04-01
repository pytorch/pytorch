//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -profile cs_6_3 -line-directive-mode none -source-embed-style binary-text -source-embed-name hello 

// CHECK: const char hello[] =
// CHECK: const size_t hello_sizeInBytes =

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// Test loop handling, with more complex structure

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
