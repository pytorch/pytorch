// snorm-modifier.slang

//TEST:COMPARE_HLSL:-profile cs_5_0 -entry main

#ifndef __SLANG__
#define input input_0
#define output output_0
#endif

Buffer<snorm float4> input;
RWStructuredBuffer<float4> output;

[numthreads(4, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    output[tid] = input[tid];
}
