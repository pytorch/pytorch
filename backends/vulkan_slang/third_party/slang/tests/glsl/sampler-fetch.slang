//TEST:SIMPLE(filecheck=CHECK): -target glsl -stage compute -entry testMain
//TEST:SIMPLE(filecheck=SPV): -target spirv -stage compute -entry testMain -emit-spirv-via-glsl

// We shouldn't be using GL_EXT_samplerless_texture_functions extension.
// CHECK-NOT: GL_EXT_samplerless_texture_functions

// SPV: OpImageFetch

Sampler2D sampler;

RWStructuredBuffer<float4> outputBuffer;

[NumThreads(1,1,1)]
void testMain()
{
    uint w, h, l;
    sampler.GetDimensions(0, w, h, l);
    var result = sampler.Load(int3(1,1,1), int2(1,2));
    outputBuffer[0] = result + float(w);
}