//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target glsl -DSLANG
//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target spirv -DSLANG

//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target glsl -DGLSL
//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target spirv -DGLSL
// CHECK: error 31212

buffer MyBlockName
{
    int result;
} outputBuffer;

#ifdef SLANG
[DerivativeGroupQuad]
[DerivativeGroupLinear]
#endif
#ifdef GLSL
layout(derivative_group_quadsNV) in;
layout(derivative_group_linearNV) in;
#endif
[numthreads(2, 2, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.result = asint(ddx(1));
}
