
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer

RWStructuredBuffer<int> outputBuffer;


int getParamSize(uint3 v)
{
    return alignof(v);
}

// On some targets inout might be passed via a pointer. 
// With "natural" layout size must be the same across all targets.
// The parameter passing semantic is "value in/value out", so the size 
// will always be the size of the *value* regardless. So uint3 in this case.
int getParamSizeWithDirection(uint3 v)
{
    return alignof(v);
}

struct TestThis : IDefaultInitializable
{
    int getSize()
    {
        return alignof(This);
    }

    float a;
    int b[2];
};

static const float4 staticConstValueA = float4(-1);

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const int idx = asint(dispatchThreadID.x);

    double doub = 13.0;
    uint16_t u16 = 10;

    uint3 vec = uint3(0);

    int size = 0;
    switch (idx)
    {   
        case 0:     size = alignof(doub); break;
        case 1:     size = alignof(u16); break;
        case 2:     size = alignof(nullptr); break;
        case 3:     size = alignof(void); break;
        
        case 4:     size = getParamSizeWithDirection(vec); break;
        case 5:     size = getParamSize(vec); break;
        
        case 6:     size = alignof(staticConstValueA); break;
        case 7:
        {
            TestThis t;
            size = t.getSize(); 
            break;
        }
        default:    size = -1; break;
    }

    outputBuffer[idx] = size;
}
