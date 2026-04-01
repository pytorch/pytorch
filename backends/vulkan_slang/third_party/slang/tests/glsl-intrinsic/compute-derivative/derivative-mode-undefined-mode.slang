//TEST:SIMPLE(filecheck=CHECK_SPV_SYNTAX): -allow-glsl -fvk-use-entrypoint-name -target spirv
// CHECK_SPV_SYNTAX: DerivativeGroupQuads{{NV|KHR}}
// CHECK_SPV_SYNTAX: DerivativeGroupQuads{{NV|KHR}}

//TEST:SIMPLE(filecheck=CHECK_GLSL_SYNTAX): -allow-glsl -stage compute -entry main2 -target glsl
// CHECK_GLSL_SYNTAX: derivative_group_quadsNV

buffer MyBlockName
{
    int result;
} outputBuffer;

[shader("compute")]
[numthreads(2, 2, 1)]
void main1(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.result = asint(ddx(1));
}

[shader("compute")]
[numthreads(2, 2, 1)]
void main2(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.result = asint(ddx(1));
}
