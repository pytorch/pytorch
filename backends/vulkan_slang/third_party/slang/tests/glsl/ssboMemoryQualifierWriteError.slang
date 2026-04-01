//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -allow-glsl -target glsl
//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -allow-glsl -target spirv -emit-spirv-directly

readonly writeonly buffer MyBlockName3
{
    uint data1;
} inputBuffer1;

writeonly buffer MyBlockName4
{
    uint data1;
} inputBuffer2;

buffer dontOptimizeOutBlock
{
    uint data1;
    uint data2;
} optimizeBlock;

layout(local_size_x = 1) in;
void computeMain()
{
//CHECK: error 30101
    optimizeBlock.data1 = inputBuffer1.data1;
//CHECK: error 30101
    optimizeBlock.data2 = inputBuffer2.data1;
//CHECK: error 30101
    int a = inputBuffer2.data1;
}