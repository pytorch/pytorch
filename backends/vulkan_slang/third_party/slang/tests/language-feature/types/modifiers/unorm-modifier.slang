// unorm-modifier.slang

//TEST:COMPARE_HLSL:-profile cs_5_0 -entry main

// Check that this compiler for glsl (without using the modifiers)
//TEST:SIMPLE(filecheck=CHECK):-target glsl -entry main -stage compute
// CHECK: uniform textureBuffer input_0;
// CHECK: texelFetch((input_0)

#ifndef __SLANG__
#define input input_0
#define output output_0
#endif


Buffer<unorm float4> input;
RWStructuredBuffer<float4> output;

[numthreads(4, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    output[tid] = input[tid];
}
