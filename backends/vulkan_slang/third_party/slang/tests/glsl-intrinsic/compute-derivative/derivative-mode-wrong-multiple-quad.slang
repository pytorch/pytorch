//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target glsl -DQUAD_SLANG
//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target spirv -DQUAD_SLANG

//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target glsl -DQUAD_GLSL
//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target spirv -DQUAD_GLSL
// CHECK: error 31210

buffer MyBlockName
{
    int result;
} outputBuffer;

#ifdef QUAD_SLANG
[DerivativeGroupQuad]
[numthreads(3, 3, 1)]
#endif
#ifdef QUAD_GLSL
layout(derivative_group_quadsNV) in;
[numthreads(3, 3, 1)]
#endif
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.result = asint(ddx(1));
}
