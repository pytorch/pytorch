// NOTE we can't test on VK/gl at the moment because we don't support intrinsics over matrices on that target currently

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -shaderobj
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Integer matrices
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

int horizontalAdd(int3 v) { return v.x + v.y + v.z; }

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);

    matrix<int, 3, 3> a = { { 0, 1, 2}, {2, 4, 6}, {16, 21, 32}};
    matrix<int, 3, 3> b = { { 4, 9, -1}, {-2, 4, 2}, {31, -3, 7}};

    matrix<int, 3, 3> t = {};
    
    t += max(a, b);
    t += min(a, b);
    t += abs(a);
    t += b % 5;

    {
        int3 low = int3(3 + idx * 1);
        int3 high = int3(5 + idx * 2);
       
        t += clamp(a, matrix<int, 3, 3>(low, low, low), matrix<int, 3, 3>(high, high, high));
    }
       
    // Access rows and elements, both read and write
   
    a[0].x ++;
    a[1][1] --;
    a[0] = a[1];
    a[2].y += b[1].x;
   
    t += a;
   
    outputBuffer[idx] = horizontalAdd(t[0]) + horizontalAdd(t[1]) + horizontalAdd(t[2]);
}
