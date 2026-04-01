//TEST:SIMPLE(filecheck=HLSL):-target hlsl -entry hullMain -stage hull -allow-glsl
//TEST:SIMPLE(filecheck=GLSL):-target glsl -entry hullMain -stage hull -allow-glsl
//TEST:SIMPLE(filecheck=SPIRV):-target spirv -entry hullMain -stage hull -allow-glsl

//HLSL: hullMain

//GLSL-DAG: location = 0
//GLSL-DAG: location = 1
//GLSL-DAG: location = 2

//SPIRV-DAG: OpDecorate {{.*}} Location 0
//SPIRV-DAG: OpDecorate {{.*}} Location 1
//SPIRV-DAG: OpDecorate {{.*}} Location 2

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
