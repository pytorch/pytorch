//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute -output-using-type -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float4> outputBuffer;

typedef float Float;

typedef vector<Float, 3> FloatVector;
typedef vector<int, 3> IntVector;
typedef vector<uint, 3> UIntVector;

void subf(inout FloatVector ft, FloatVector f, int idx, Float vf)
{

    ft += log(f + 10.0);
    ft += log2(f * 3 + 2);

    {
        float v[] = { 1, 10, 100, 1000 };
        ft += IntVector(log10(FloatVector(v[idx] + vf) + 0.5f));
    }

    ft += abs(f * 4 - 2.0f);

    ft += min(0.5, f);
    ft += max(f, 0.75);

    ft += pow(0.5, f);

    ft += smoothstep(0.2, 0.7, f);
    ft += lerp(-100, 100, f);

    ft += clamp(f, 0.1, 0.3);

    ft += step(f, 0.5);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    Float vf = idx * (1.0f / (4.0f));
     
    FloatVector f = FloatVector(0.1f, vf, vf + 0.2f);
    
    // Operate over all values
    FloatVector ft = {};

    // fmod
    ft += FloatVector(IntVector(((f % 0.11f) * 100) + 0.5));
    
    ft += sin(f);
    ft += cos(f);
    ft += tan(f);
    
    {
        // Vector specific
        FloatVector nv = normalize(f);
        ft += nv;
        
        FloatVector perp = cross(nv, f.zxy); 
        ft += perp;
  
        ft += dot(perp, f.zyx);
        ft += length(perp);
  
        ft += reflect(f, perp.yzx);
    }
    
    ft += asin(f);
    ft += acos(f);
    ft += atan(f);
    
    ft += atan2(f, 2.0);

#if 0
    {
        // Disabled because not supported on VK (glsl) in vector form
        FloatVector sf, cf;
        sincos(f, sf, cf);
        
        ft += sf;
        ft += cf;
    }
#endif     
     
#if 0     
    // Disabled because not supported on VK (glsl) in vector form
    ft += rcp(1.0 + f);
#endif

    ft += sign(f - 0.5);
    
    ft += saturate(f * 4 - 2.0);
    
    ft += sqrt(f);
    ft += rsqrt(1.0f + f);
    
    ft += exp2(f);
    ft += exp(f);
     
    ft += frac(f * 3);
    ft += ceil(f * 5 - 3);
    
    ft += floor(f * 10 - 7.01);

    ft += trunc(f * 7);

    subf(ft, f, idx, vf);

    {
        IntVector vi = asint(f - f) + idx;
        ft += FloatVector(vi);   
    }
    
    outputBuffer[idx] = vector<Float, 4>(ft, 0);
}
