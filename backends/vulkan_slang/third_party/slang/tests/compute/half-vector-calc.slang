//DISABLE_TEST(compute):COMPARE_COMPUTE:-dx12 -compute -output-using-type -use-dxil -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -compute -output-using-type -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -compute -output-using-type -render-features half -shaderobj

// Test for doing a calculation using half

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int x = tid.x;
    
    half2 v3 = half2(float(x));
    
    half2 v0 = half2(float2(x * 2.0f, x * 0.5f));
    half3 v1 = half3(half(x * 2.0f), half(x * 0.5f), half(x - 1.0f));
    half4 v2 = half4(half(x + 1), half(x - 1), half(x + 2) , half(x - 2));
    
    v1 += v0.yxy;
    v1 += v2.wzy;
    v2 += v0.xyxy;
    
    v1 ++;
    --v2;
    v3++;
    
    // Unary
    v2 = +v2.yxwz;
    v2.xyz = -v2.zwx;
    
    // Scalar vector 
    v1 = v1 + v2.x;
    v2 = v2 * half(2.0f);
    v0 = half(2.0f) * v0;
    v2 = v2 / half(2.0f);
    
    v0 *= half(2.0f);
    
    v0 = v0 + v0 * v0;
    v1 = v1 + v1 * v1;
    v2 = v2 + v2 * v2;
    
    half o2 = v2.x + v2.y + v2.z + v2.w;
    half o1 = v1.x + v1.y + v1.z;
    half o0 = v0.x + v0.y;
        
    outputBuffer[tid] = o0 + o1 + o2 + v3.y;
}
