//DISABLE_TEST:SIMPLE:-target dxil -entry computeMain -profile cs_6_2

struct ValueContainer
{
    Texture2D<uint> values;
 
    uint getValue(uint2 at)
    {
        return values[at];
    }
};
  
ParameterBlock<ValueContainer> gContainer;

Texture2D<uint> gValues;    
    
RWStructuredBuffer<int> outputBuffer;    
    
[numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    ValueContainer container;

    // This will work
    // container = gContainer;
    // Or just setting directly
    // container.values = gValues; 
    
    // But if values in container is not .... then specialization fails
    // and invalid HLSL is output.
    
    uint2 pos = dispatchThreadID.xy;
    
    outputBuffer[pos.x + pos.y] = container.getValue(pos);
}

