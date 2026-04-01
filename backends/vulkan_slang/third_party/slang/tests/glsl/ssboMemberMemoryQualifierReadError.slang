//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -allow-glsl -target glsl
//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -allow-glsl -target spirv -emit-spirv-directly

volatile buffer MyBlockName1
{
    readonly uint data2;
    readonly writeonly uint data3;
} inputBuffer1;

layout(local_size_x = 1) in;
void computeMain()
{
//CHECK: error 30011
    inputBuffer1.data2 = 1;
//CHECK: error 30011
    inputBuffer1.data3 = 1;
}