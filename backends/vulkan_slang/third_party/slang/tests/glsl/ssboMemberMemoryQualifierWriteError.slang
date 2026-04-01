//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -allow-glsl -target glsl
//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -allow-glsl -target spirv -emit-spirv-directly

volatile buffer MyBlockName1
{
    writeonly uint data2;
    readonly writeonly uint data3;
} inputBuffer1;

buffer MyBlockName2
{
    uint data;
} outputBuffer;

layout(local_size_x = 1) in;
void computeMain()
{
//CHECK: error 30101
    int a = inputBuffer1.data2;
//CHECK: error 30101
    int b = inputBuffer1.data3;
//CHECK: error 30101
    outputBuffer.data = inputBuffer1.data2;
}