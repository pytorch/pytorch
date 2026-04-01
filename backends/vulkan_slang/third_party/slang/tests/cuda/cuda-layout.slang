//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer : register(u0);

#define WRITE_TYPE_ALIGN(base, type) \
    outputBuffer[base * 4 + 0] = __alignOf<type>(); \
    outputBuffer[base * 4 + 1] = __alignOf<vector<type, 2> >(); \
    outputBuffer[base * 4 + 2] = __alignOf<vector<type, 3> >(); \
    outputBuffer[base * 4 + 3] = __alignOf<vector<type, 4> >();
    

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    WRITE_TYPE_ALIGN(0, uint8_t)
    WRITE_TYPE_ALIGN(1, uint16_t)
    WRITE_TYPE_ALIGN(2, int)
    WRITE_TYPE_ALIGN(3, int64_t)
    
    WRITE_TYPE_ALIGN(4, half)
    WRITE_TYPE_ALIGN(5, float)
    WRITE_TYPE_ALIGN(6, double)   
}
