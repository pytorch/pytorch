//TEST:SIMPLE(filecheck=GLSL):-target glsl -entry hullMain -stage hull -allow-glsl

// Currently SPIR-V fails to compile this hull shader (#4914)
//DISABLE_TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry hullMain -stage hull -allow-glsl

//GLSL-DAG: location = 0
//GLSL-DAG: location = 1
//GLSL-DAG: location = 2
//GLSL-DAG: location = 3

//SPIRV-DAG: location 0
//SPIRV-DAG: location 1
//SPIRV-DAG: location 2
//SPIRV-DAG: location 3


struct HsOut
{
    float2 pos;
    float2 hm;
};

struct HscOut
{
    uint roundingFactor;
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
    o.roundingFactor = 6;
    o.EdgeTessFactor[0] = 1;
    o.EdgeTessFactor[1] = 2;
    o.EdgeTessFactor[2] = 3;
    o.EdgeTessFactor[3] = 4;
    o.InsideTessFactor[0] = 0.5;
    o.InsideTessFactor[1] = 0.5;
    return o;
}