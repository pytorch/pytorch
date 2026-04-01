//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    float f = idx * (1.0f / (4.0f));

    float ft = 0.0f;
    
    // fmod
    ft += int(((f % 0.11f) * 100) + 0.5);
    
    ft += sin(f);
    ft += cos(f);
    ft += tan(f);
    
    ft += asin(f);
    ft += acos(f);
    ft += atan(f);
    
    ft += length(f - 0.5);
    
    ft += atan2(f, 2.0);
    
    {
        float sf, cf;
        sincos(f, sf, cf);
        
        ft += sf;
        ft += cf;
    }
    
    ft += rcp(1.0 + f);
    ft += sign(f - 0.5);
    
    ft += saturate(f * 4 - 2.0);
    
    ft += sqrt(f);
    ft += rsqrt(1.0f + f);
    
    ft += exp2(f);
    ft += exp(f);
        
        
    ft += frac(f * 3);
    ft += ceil(f * 5 - 3);
    
    ft += floor(f * 10 - 7);
    ft += trunc(f * 7);
     
   
    ft += log(f + 10.0);
    ft += log2(f * 3 + 2);

    {
        float v[] = { 1, 10, 100, 1000 };
        ft += int(log10(v[idx]) + 0.5f);
    }
       
    ft += abs(f * 4 - 2.0f);
    
    ft += min(0.5, f);
    ft += max(f, 0.75);

    ft += pow(0.5, f);

    ft += smoothstep(0.2, 0.7, f);
    ft += lerp(-100, 100, f);


    ft += clamp(f, 0.1, 0.3);

    ft += step(f, 0.5);

    int vi = asint(f - f) + idx;
    
    ft += float(vi);

    ft += ldexp(23.2, f);

    uint vu = asuint(f);
    ft += asfloat(vu);
    
    outputBuffer[idx] = int(ft * 16);
}
