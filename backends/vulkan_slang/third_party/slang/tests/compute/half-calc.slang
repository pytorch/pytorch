//DISABLE_TEST(compute):COMPARE_COMPUTE:-dx12 -compute -use-dxil -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -compute -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -compute -render-features half -shaderobj

// Test for doing a calculation using half

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    
    //half2 v0 = { 1, -2 };
    //half2 v1 = { -2, 4 };
    
    //half2 v2 = (v0 * 2.0f + v1);
    
    // This should work (it compiles on dxc, but slang doesn't seem to have overloads yet)
    //half offset = length(v2);
    
    //half offset = v2.x + v2.y;
    
    half offset = half(0.0f);
    
    half v = half(tid);
    v *= half(3.0f);
    v += half(1.0f);
    v += offset;
    
    v++;
    --v;
    v--;
    
    outputBuffer[tid] = v;
}