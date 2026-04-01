// func-resource-result-simple.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
// Not supported in WGSL: Arrays of textures or buffers
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

// Test that a function that returns a resource type can be
// compiled for targets that don't natively support resource
// return values.

//TEST_INPUT:set textures=[Texture2D(size=4, content = zero), Texture2D(size=4, content = one)]
Texture2D textures[2];

//TEST_INPUT:set sampler=Sampler
SamplerState sampler;

Texture2D getTex(int index)
{
    Texture2D result;
    // Note: `index` here will need to be a compile time constant in order to generate
    // valid GLSL. If constant folding and function inlining are all done correctly
    // we should be able to compile this.
    if (index == 0)
        result = textures[1];
    else
        result = textures[0];
    return result;
}

int test(int val)
{
    // Make sure index is a compile-time constant.
	return getTex(int(0.0) + 1*2 < 5 ? 0 : 1).SampleLevel(sampler, float2(0,0), 0).x == 0.0 ? 0 : 1;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
