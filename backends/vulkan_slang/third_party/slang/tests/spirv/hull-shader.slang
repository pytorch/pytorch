//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry hullMain -stage hull -allow-glsl

// SPIRV: OpDecorate %entryPointParam_hullMain_pos Location 0
// SPIRV: OpDecorate %entryPointParam_hullMain_hm Location 1
// SPIRV: OpDecorate %entryPointParam_constants_instanceId Location 2

struct HsOut
{
    float2 pos;
    float2 hm;
};

struct HscOut
{
    float EdgeTessFactor[4] : SV_TessFactor;
    float InsideTessFactor[2] : SV_InsideTessFactor;
    uint instanceId;
};

[domain("quad")]
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
    o.instanceId = 123;
    o.EdgeTessFactor[0] = 1;
    o.EdgeTessFactor[1] = 2;
    o.EdgeTessFactor[2] = 3;
    o.EdgeTessFactor[3] = 4;
    o.InsideTessFactor[0] = 0.5;
    o.InsideTessFactor[1] = 0.5;
    return o;
}