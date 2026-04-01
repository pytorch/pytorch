//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target glsl -DLINEAR_SLANG
//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target spirv -DLINEAR_SLANG

//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target glsl -DLINEAR_GLSL
//TEST:SIMPLE(filecheck=CHECK): -allow-glsl -stage compute -entry computeMain -target spirv -DLINEAR_GLSL
// CHECK: error 31211

buffer MyBlockName
{
    int result;
} outputBuffer;

#ifdef LINEAR_SLANG
[DerivativeGroupLinear]
[numthreads(1, 3, 1)]
#endif

#ifdef LINEAR_GLSL
layout(derivative_group_linearNV) in;
[numthreads(1, 3, 1)]
#endif

void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.result = asint(ddx(1));
}
