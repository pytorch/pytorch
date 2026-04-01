//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-mtl -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int id = asint(dispatchThreadID.x);

    {
        int4 i4 = int4(id, id + 1, id + 2, id + 3);
        
        // Identity - int vector
        i4 = asint(i4);
        
        uint3 u3 = asuint(i4.xyz);
        // Identity  - uint vec
        u3 = asuint(u3);
        
        int2 i2 = asint(u3.xy);
        uint u1 = asuint(i2.x);
        
        // Identity - uint scalar
        u1 = asuint(u1);
        
        outputBuffer[id + 0] = int(u1);
    }
    
    {
        uint v = asuint(id);
        uint4 u4 = uint4(v, v + 1, v + 2, v + 3);
        int3 i3 = asint(u4.xyz);
        uint2 u2 = asuint(i3.xy);
        int i1 = asint(u2.x);
        
        // Identity - int scalar
        i1 = asint(i1);
        
        outputBuffer[id + 4] = i1;
    }
    
    {
        // Make i4 holds id as floats so we know they are valid float values and not denormals
        int4 i4 = int4(asint(float(id)), asint(float(id + 1)), asint(float(id + 2)), asint(float(id + 3))); 
        float4 f4 = asfloat(asfloat(i4));
        
        // Identity - float vector
        f4 = asfloat(f4);
        
        uint3 u3 = asuint(f4.xyz);
        float2 f2 = asfloat(u3.xy);
        uint u1 = asuint(f2.x);
        
        float f1 = asfloat(asfloat(u1));
        
        // Identity - float scalar
        f1 = asfloat(f1);
        
        int i1 = asint(f1);
        
        float f1_ = asfloat(i1);
        int i1_ = asint(f1_);
        
        outputBuffer[id + 8] = (int)asfloat(i1_); 
    }
}
