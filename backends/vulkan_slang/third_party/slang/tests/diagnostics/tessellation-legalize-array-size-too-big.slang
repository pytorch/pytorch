//TEST:SIMPLE(filecheck=CHK):-target glsl -entry hullMain -stage hull -allow-glsl
//TEST:SIMPLE(filecheck=CHK):-target spirv-asm -entry hullMain -stage hull -allow-glsl

// Tessllation outsie factor must be an array of size 4 or less
// Tessllation inside factor must be an array of size 2 or less

struct HsOut
{
    float2 pos;
    float2 hm;
};

struct HscOut
{
    //CHK: error 30024: Cannot convert array of size 5 to array of size 4 as this would truncate data
    float EdgeTessFactor[5] : SV_TessFactor;

    //CHK: error 30024: Cannot convert array of size 3 to array of size 2 as this would truncate data
    float InsideTessFactor[3] : SV_InsideTessFactor;
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
    o.EdgeTessFactor[3] = 4;
    o.EdgeTessFactor[4] = 5;
    o.InsideTessFactor[0] = 0.5;
    o.InsideTessFactor[1] = 0.3;
    o.InsideTessFactor[2] = 0.2;
    return o;
}
