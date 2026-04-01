// enum.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

// Confirm operations that produce bools - such as comparisons, or && ||, ! work correctly

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(16, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;; 

    uint uv = tid;
    int iv = int(tid);
    
    bool2 bv2 = { (tid & 1) != 0, tid > 10 };      
    
    float2 f2 = { float(tid), float(tid + 1) };
    
    bool bv = tid > 6;
    let not_bv2 = !bv2;
    
    int r = 0;
    if (bv)
    {
        r |= 0x0001;
    }
    if (!bv)
    {
        r |= 0x0002;
    }
    
    if (iv != 0)
    {
        r|= 0x0004;
    }
    if (!bool(iv))
    {
        r|= 0x0008;
    }

    if (uv != 0)
    {
        r |= 0x0010;
    }
    if (!bool(uv))
    {
        r |= 0x0020;
    }
    
    if (all(bv2))
    {
        r |= 0x0040;
    }
    if (all(not_bv2))
    {
        r |= 0x0080;
    }
    
    if (any(!bool2(f2)))
    {
        r |= 0x0100;
    }
    
    // TODO(JS): Support on GLSL targets
    // This doesn't currently work on GLSL targets, and because there 
    // do not appear to be any vector bitwise operations. Could be achieved
    // by deconstructing, and reconstructing the vec result
    /* if (all(f2 || bv2))
    {
        r |= 0x0200;
    } */
    
    outputBuffer[tid] = r;
}