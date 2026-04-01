//TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
#version 460
// CHECK: warning 31200

buffer MyBlockName2
{
    uint data[];
} outputBuffer;

layout(local_size_x = 4) in;

bool testNoiseScalar()
{
    return true
        && noise1(1.0f) == 0.0f
        && noise2(1.0f) == vec2(0.0f)
        && noise3(1.0f) == vec3(0.0f)
        && noise4(1.0f) == vec4(0.0f)
        ;
}

__generic<let N:int>
bool testNoiseVector()
{
    return true
        && noise1(vector<float, N>(1.0f)) == 0.0f
        && noise2(vector<float, N>(1.0f)) == vec2(0.0f)
        && noise3(vector<float, N>(1.0f)) == vec3(0.0f)
        && noise4(vector<float, N>(1.0f)) == vec4(0.0f)
        ;
}

bool testNoiseFunctions()
{
    return true
        && testNoiseScalar()
        && testNoiseVector<2>()
        && testNoiseVector<3>()
        && testNoiseVector<4>()
        ;
}

void computeMain()
{
    outputBuffer.data[0] = true
        && testNoiseFunctions()
        ;
    // BUF: 1
}