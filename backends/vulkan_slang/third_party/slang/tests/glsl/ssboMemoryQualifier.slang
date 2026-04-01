//TEST:SIMPLE(filecheck=CHECK_GLSL): -stage compute -entry computeMain -allow-glsl -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -stage compute -entry computeMain -allow-glsl -target spirv -emit-spirv-directly

// CHECK_SPV: OpEntryPoint

//CHECK_GLSL: volatile
//CHECK_SPV: OpDecorate %{{.*}} Volatile
volatile buffer MyBlockName1
{
    uint data1;
} inputBuffer1;

//CHECK_GLSL: restrict
//CHECK_SPV: OpDecorate %{{.*}} Restrict
restrict buffer MyBlockName2
{
    uint data1;
} inputBuffer2;

//CHECK_GLSL: {{(writeonly|volatile)}} {{(writeonly|volatile)}}
//CHECK_SPV: OpDecorate %{{.*}} {{(NonReadable|Volatile)}}
//CHECK_SPV: OpDecorate %{{.*}} {{(NonReadable|Volatile)}}
writeonly volatile buffer MyBlockName3
{
    uint data1;
} inputBuffer3;

//CHECK_GLSL: writeonly
//CHECK_SPV: OpDecorate %{{.*}} NonReadable
writeonly buffer MyBlockName4
{
    uint data1;
} inputBuffer4;

//CHECK_GLSL: readonly
//CHECK_SPV: OpDecorate %{{.*}} NonWritable
readonly buffer MyBlockName5
{
    uint data1;
} inputBuffer5;

//CHECK_GLSL: coherent
//CHECK_SPV: OpDecorate %{{.*}} Coherent
coherent buffer MyBlockName6
{
    uint data1;
} inputBuffer6;

// CHECK_GLSL: main
layout(local_size_x = 1) in;
void computeMain()
{
    inputBuffer1.data1 = 1;
    inputBuffer2.data1 = 1;
    inputBuffer3.data1 = 1;
    inputBuffer4.data1 = 1;
    int v = inputBuffer5.data1;
    // ensure code is not optimized out
    inputBuffer6.data1 = v;
}