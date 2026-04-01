//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    uint data[1];
} outputBuffer;

layout(local_size_x = 1) in;

//TEST_INPUT: set image_1d = RWTexture1D(format=R32Sint, size=4, content=one, mipMaps = 1)
uniform layout(binding=0,r32i) iimage1D image_1d;

bool checkAllImageAtomicAdd()
{
    imageStore(image_1d, 0, ivec4(1));
    return true
        && imageAtomicAdd(image_1d, 0, 0) == 1
        && imageLoad(image_1d, 0).x == 1
        ;
}

// CHECK_GLSL: void main(
// CHECK_SPV: OpEntryPoint
void computeMain()
{
    outputBuffer.data[0] = true
        && checkAllImageAtomicAdd()
        ;
    // BUF: 1
}
