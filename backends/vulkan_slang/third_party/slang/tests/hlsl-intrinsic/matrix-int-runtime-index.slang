// NOTE we can't test on VK/gl because we will output imat3 - and imat3 is not supported by glsl.
// NOTE we also can't do this test on fxc - as it does not allow runtime non constant indexing.

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -shaderobj
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

int horizontalAdd(int3 v) { return v.x + v.y + v.z; }

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    matrix<int, 3, 3> a = { { 0, 1, 2}, {2, 4, 6}, {7, 21, 32}};
    matrix<int, 3, 3> b = { { 4, 9, -1}, {-2, 4, 2}, {31, -3, 7}};

    // Do a dynamic row swap
    {
        matrix<int, 3, 3> c = a;
    
        int row0 = idx % 3;
        int row1 = (idx + 1) % 3; 
        int row2 = (idx + 2) % 3;
        
        a[row0] = c[0];
        a[row1] = c[1];
        a[row2] = c[2];
    }
 
    outputBuffer[idx] = (a[0].x & 15) + ((a[1].x & 15) << 4) + ((a[2].x & 15) << 8);
}