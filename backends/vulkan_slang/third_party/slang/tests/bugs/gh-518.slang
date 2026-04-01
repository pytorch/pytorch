//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST_DISABLED(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
// Not supported in WGSL: Arrays of textures or buffers
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

// Note: can't actually test this on Vulkan right now because
// the support in render-test isn't good enough.

// Confirm that we can handle arrays of resources
// being passed to a function.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name gBuffer
RWStructuredBuffer<int> gBuffer;

//TEST_INPUT: Texture2D(size=4, content=one):name gTextures[0]
//TEST_INPUT: Texture2D(size=4, content=one):name gTextures[1]
Texture2D gTextures[2];

float broken(Texture2D t[2])
{
    return t[0].Load(int3(0)).x + t[1].Load(int3(0)).x;
}

int test(int val)
{
    return val*int(broken(gTextures));
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    int val = int(tid);
    val = test(val);
    gBuffer[tid] = val;
}