//DISABLE_TEST(compute):COMPARE_COMPUTE:-dx12 -compute -output-using-type -use-dxil -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -compute -output-using-type -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -compute -output-using-type -render-features half -shaderobj

// Test for doing a calculation using half

//TEST_INPUT:ubuffer(data=[0.2 10.0 12.0 16.0], stride=4):name=inputBuffer
RWStructuredBuffer<int> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct Values
{
    __init(int index)
    {
        m_index = index;
    }
    
    [mutating] half next()
    {
        float v = inputBuffer[m_index & 3];
        m_index++;
        return half(v);
    }
    
    int m_index = 0;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    
    Values values = Values(int(tid));
    
    int r = 0;
    
    half s0 = values.next();
    half s1 = values.next();
    
    if (s0 < s1)
    {
        r += 0x1;
    }
    
    half2 h2_0 = half2(values.next(), values.next());
    half2 h2_1 = half2(values.next(), values.next());
    
    if (any(h2_0 < h2_1))
    {
        r += 0x2;
    }
    
    if (all(h2_0 < h2_1))
    {
        r += 0x4;
    }
    
    half3 h3_0 = half3(values.next(), values.next(), values.next());
    half3 h3_1 = half3(values.next(), values.next(), values.next());
    
    if (any(h3_0 > h3_1))
    {
        r += 0x8;
    }
    
    if (all(h3_0 <= h3_1))
    {
        r += 0x10;
    }
    
    half4 h4_0 = half4(values.next(), values.next(), values.next(), values.next());
    half4 h4_1 = half4(values.next(), values.next(), values.next(), values.next());
    
    
    if (any(h4_0 > h4_1))
    {
        r += 0x8;
    }
    
    if (all(h4_0 <= h4_1))
    {
        r += 0x10;
    }
    
    if (any(!(h4_0 == h4_1)))
    {
        r += 0x20;
    }
    
    if (all(h4_0 != h4_1))
    {
        r += 0x40;
    }
    
    outputBuffer[tid] = r;
}
