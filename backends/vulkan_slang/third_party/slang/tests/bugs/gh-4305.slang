//TEST:SIMPLE(filecheck=CHECK): -stage compute  -entry computeMain -target glsl -allow-glsl -DQUAD -DMODIFIER
//TEST:SIMPLE(filecheck=CHECK): -stage compute  -entry computeMain -target glsl -allow-glsl -DMODIFIER
//TEST:SIMPLE(filecheck=CHECK): -stage compute  -entry computeMain -target glsl -allow-glsl -DQUAD
//TEST:SIMPLE(filecheck=CHECK): -stage compute  -entry computeMain -target glsl -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -output-using-type -emit-spirv-via-glsl -allow-glsl -xslang -DQUAD
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -output-using-type -emit-spirv-via-glsl -allow-glsl

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;
//TEST_INPUT: Texture2D(size=4, content = one):name t2D
Texture2D<float> t2D;
//TEST_INPUT: Sampler:name samplerState
SamplerState samplerState;

// "local_size_x" must appear before "derivative_group_"
//CHECK:local_size_x
//CHECK:derivative_group_{{.*}}NV

#ifndef MODIFIER
#ifdef QUAD
layout(derivative_group_quadsNV) in;
#else
layout(derivative_group_linearNV) in;
#endif // #ifdef QUAD
layout(local_size_x = 2, local_size_y = 2, local_size_z = 1) in;

#else

#ifdef QUAD
[DerivativeGroupQuad]
#else
[DerivativeGroupLinear]
#endif // #ifdef QUAD
[numthreads(2, 2, 1)]
#endif //#ifndef MODIFIER
void computeMain()
{
    //BUF:1
    outputBuffer[0] = t2D.Sample(samplerState, float2(0.5, 0.5));
}