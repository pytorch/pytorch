//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4,4,1)]
void computeMain(int2 pixelIndex : SV_DispatchThreadID)
{
    
    // We will test floats, uints, and int vectors
    // We need all comparisons < > <= >= == !=
    
    int x = pixelIndex.x;
    int y = pixelIndex.y;
    
    int uintBits;
    {
        uint2 v = { uint(x), uint(y) };
        uint2 test = { 1, 3 };
        
        int bits = 0;
        bits |= any(v <  test)   ? 0x01 : 0;
        bits |= any(v <= test)   ? 0x02 : 0;
        bits |= any(v == test)   ? 0x04 : 0; 
        bits |= any(v >  test)   ? 0x08 : 0;
        bits |= any(v >= test)   ? 0x10 : 0;
        bits |= any(v != test)   ? 0x20 : 0;
        uintBits = bits;
    }
    
    int intBits;
    {
        int2 v = { x, y };
        int2 test = { 2, 0 };
        
        int bits = 0;
        bits |= any(v <  test)   ? 0x01 : 0;
        bits |= any(v <= test)   ? 0x02 : 0;
        bits |= any(v == test)   ? 0x04 : 0; 
        bits |= any(v >  test)   ? 0x08 : 0;
        bits |= any(v >= test)   ? 0x10 : 0;
        bits |= any(v != test)   ? 0x20 : 0;
        intBits = bits;
    }
    
    int floatBits;
    {
        float2 v = { float(x), float(y) };
        float2 test = { 1, 2 };
        
        int bits = 0;
        bits |= any(v <  test)   ? 0x01 : 0;
        bits |= any(v <= test)   ? 0x02 : 0;
        bits |= any(v == test)   ? 0x04 : 0; 
        bits |= any(v >  test)   ? 0x08 : 0;
        bits |= any(v >= test)   ? 0x10 : 0;
        bits |= any(v != test)   ? 0x20 : 0;
        floatBits = bits;
    }
       
    int coerceBits;
    {
        float2 v = { float(x), float(y) };
        int2 test = { 1, 2 };
        
        int bits = 0;
        bits |= any(v <  test)   ? 0x01 : 0;
        bits |= any(v <= test)   ? 0x02 : 0;
        bits |= any(v == test)   ? 0x04 : 0; 
        bits |= any(v >  test)   ? 0x08 : 0;
        bits |= any(v >= test)   ? 0x10 : 0;
        bits |= any(v != test)   ? 0x20 : 0;
        coerceBits = bits;
    }
    
    outputBuffer[pixelIndex.x + pixelIndex.y * 4] = (coerceBits << 24) | (floatBits << 16) | (intBits << 8) | (uintBits);
}
