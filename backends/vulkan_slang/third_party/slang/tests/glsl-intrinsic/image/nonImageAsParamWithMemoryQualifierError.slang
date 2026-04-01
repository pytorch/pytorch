//TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer

// restrict modifier is allowed to be dropped as a parameter
buffer MyBlockName2
{
    restrict uint data[1];
} outputBuffer;

layout(local_size_x = 4) in;

// CHECK: error 31206
bool checkAllImageSizes(writeonly uint val)
{
    return true
        && val == 0
        ;
}

void computeMain()
{
    outputBuffer.data[0] = true
        && checkAllImageSizes(outputBuffer.data[0])
        ;
}