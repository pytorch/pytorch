//TEST(compute):COMPARE_COMPUTE:-cpu
//TEST(compute):COMPARE_COMPUTE: -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Arrays of textures or buffers
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
//TEST_INPUT:ubuffer(data=[0 2 3 3]):name=s[0].a
//TEST_INPUT:ubuffer(data=[4 5 6 7]):name=s[1].a
//TEST_INPUT:ubuffer(data=[8 9 10 11]):name=s[0].b
//TEST_INPUT:ubuffer(data=[12 13 14 15]):name=s[1].b

RWStructuredBuffer<int> outputBuffer;

struct S
{
	RWByteAddressBuffer a;
	RWByteAddressBuffer b;
};
S s[2];

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int i = dispatchThreadID.x;

    int val = int(
          s[0].a.Load(i * 4)
        + s[1].a.Load(i * 4)*16
        + s[0].b.Load(i * 4)*256
        + s[1].b.Load(i * 4)*4096);

    outputBuffer[i] = val; 
}
