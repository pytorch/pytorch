//TEST:SIMPLE(filecheck=CHECK_SPV_SYNTAX): -allow-glsl -fvk-use-entrypoint-name -target spirv
// CHECK_SPV_SYNTAX: DerivativeGroupQuads{{NV|KHR}}
// CHECK_SPV_SYNTAX: DerivativeGroupLinear{{NV|KHR}}

//TEST:SIMPLE(filecheck=CHECK_GLSL_SYNTAX): -allow-glsl -fvk-use-entrypoint-name -target spirv -DGLSL_SYNTAX
// CHECK_GLSL_SYNTAX: DerivativeGroupQuads{{NV|KHR}}
// CHECK_GLSL_SYNTAX: DerivativeGroupQuads{{NV|KHR}}
#ifdef GLSL_SYNTAX
layout(derivative_group_quadsNV) in;
#endif

buffer MyBlockName
{
    int result;
} outputBuffer;

[DerivativeGroupQuad]
[shader("compute")]
[numthreads(2,2,1)]
void main1(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.result = asint(ddx(1));
}

[DerivativeGroupLinear]
[shader("compute")]
void main2(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.result = asint(ddx(1));
}
