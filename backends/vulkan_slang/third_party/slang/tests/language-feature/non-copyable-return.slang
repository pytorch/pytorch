//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):SIMPLE(filecheck=GLSL): -stage compute -entry computeMain -target glsl

// Note: spirv_by_reference is only supported for passing opaque types, so this test won't produce
// expected result on vulkan.
//DISABLED_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

[__NonCopyableType]
struct MyType
{
    float x;
    __init() { x = 1.0; }
}

MyType myFunc1(float y)
{
    __return_val = MyType();
    __return_val.x += y;
}

MyType myFunc0(float x)
{
    return myFunc1(x + 1.0);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    let f = myFunc0(2.0);
    // CHECK: 4.0
    // GLSL: main(
    // GLSL-NOT: MyType {{.*}} =
    outputBuffer[0] = f.x;
}
