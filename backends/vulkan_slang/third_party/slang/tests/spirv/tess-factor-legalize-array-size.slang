//TEST:SIMPLE(filecheck=GLSL):-target glsl -entry hullMain -stage hull -allow-glsl
//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry hullMain -stage hull -allow-glsl

// When targeting SPIRV, the tessellation factors should be an array with size 4 or 2.

// GLSL: gl_TessLevelOuter[3] = 0.0;
// GLSL: gl_TessLevelInner[1] = 0.0;

// SPIRV: OpDecorate %gl_TessLevelOuter BuiltIn TessLevelOuter
// SPIRV: OpDecorate %gl_TessLevelInner BuiltIn TessLevelInner

// SPIRV-DAG: OpStore %gl_TessLevelOuter %[[OuterOpCompositeConstruct:[1-9][0-9]*]]
// SPIRV-DAG: %[[OuterOpCompositeConstruct]] = OpCompositeConstruct %[[OuterOpTypeArray:[a-zA-Z_0-9]*]]
// SPIRV-DAG: %[[OuterOpTypeArray]] = OpTypeArray %{{[^ ]*}} %[[OuterOpConstant:[a-zA-Z_0-9]*]]
// SPIRV-DAG: %[[OuterOpConstant]] = OpConstant %{{[^ ]*}} 4

// SPIRV-DAG: OpStore %gl_TessLevelInner %[[InnerOpCompositeConstruct:[1-9][0-9]*]]
// SPIRV-DAG: %[[InnerOpCompositeConstruct]] = OpCompositeConstruct %[[InnerOpTypeArray:[a-zA-Z_0-9]*]]
// SPIRV-DAG: %[[InnerOpTypeArray]] = OpTypeArray %{{[^ ]*}} %[[InnerOpConstant:[a-zA-Z_0-9]*]]
// SPIRV-DAG: %[[InnerOpConstant]] = OpConstant %{{[^ ]*}} 2

struct HsOut
{
    float2 pos;
    float2 hm;
};

struct HscOut
{
    float EdgeTessFactor[3] : SV_TessFactor; // Should be legalized to float[4]
    float InsideTessFactor[1] : SV_InsideTessFactor; // Should be legalized to float[2]
};

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_ccw")]
[outputcontrolpoints(4)]
[patchconstantfunc("constants")]
HsOut hullMain()
{
    HsOut o;
    o.pos = 1;
    o.hm = 2;
    return o;
}

HscOut constants()
{
    HscOut o;
    o.EdgeTessFactor[0] = 1;
    o.EdgeTessFactor[1] = 2;
    o.EdgeTessFactor[2] = 3;
    o.InsideTessFactor[0] = 0.5;
    return o;
}
