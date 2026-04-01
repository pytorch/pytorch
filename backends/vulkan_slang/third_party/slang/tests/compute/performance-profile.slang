//TEST(compute):PERFORMANCE_PROFILE:-cpu -compute -compile-arg -O3 -compute-dispatch 256,1,1  -shaderobj
//TEST_DISABLED(compute):PERFORMANCE_PROFILE:-cpu -compute -source-language cpp -compile-arg -O3 -compute-dispatch 256,1,1
//TEST(compute):PERFORMANCE_PROFILE:-slang -compute -compute-dispatch 256,1,1 -shaderobj
//TEST(compute):PERFORMANCE_PROFILE:-slang -compute -dx12 -compute-dispatch 256,1,1 -shaderobj
//TEST(compute, vulkan):PERFORMANCE_PROFILE:-vk -compute -compute-dispatch 256,1,1 -shaderobj

//TEST_INPUT:ubuffer(random(float, 4096, -1, 1), stride=4):out,name outputBuffer

#ifndef __cplusplus

RWStructuredBuffer<float> outputBuffer;

[numthreads(16, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint i = dispatchThreadID.x;
    float v = outputBuffer[i];
    v = v < 0.0f ? (v * v) : (v + v + v);
	outputBuffer[i] = v;
}


#else

namespace { // anonymous

struct LocalUniformState
{
    RWStructuredBuffer<float> outputBuffer_0;
};

} // anonymous

static void _calc(const RWStructuredBuffer<float>& buf, int start, int end)
{
    assert(start >= 0 && end <= buf.count);
    float* data = buf.data;
    
    for (int i = start; i < end; ++i)
    {
        float v = data[i];
        data[i] = v < 0.0f ? (v * v) : (v + v + v);
    }
}

SLANG_PRELUDE_EXPORT
void computeMain(ComputeVaryingInput* varyingInput, void* inParams, void* inUniformState)
{
    LocalUniformState* uniformState = (LocalUniformState*)inUniformState;
    _calc(uniformState->outputBuffer_0, varyingInput->startGroupID.x * 16, varyingInput->endGroupID.x * 16);
}
 
#endif