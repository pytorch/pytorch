//TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    uint data[1];
} outputBuffer;

layout(local_size_x = 4) in;

//TEST_INPUT: set someImage = RWTexture2D(format=RGBA16Float, size=1, content=one, mipMaps = 1)
uniform layout(binding=0,rgba16f) readonly image2D someImage;

// CHECK: error 30048
bool checkAllImageSizes(writeonly image2D val)
{
    return true
        && imageSize(val) == ivec2(1)
        ;
}

void computeMain()
{
    outputBuffer.data[0] = true
        && checkAllImageSizes(someImage);
        ;
}
