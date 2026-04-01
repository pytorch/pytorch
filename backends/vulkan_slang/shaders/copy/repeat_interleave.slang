// repeat_interleave: each output element maps to input[tid / repeats]
// Input is flattened, output has repeats * input.numel() elements.
// For dim variant: input is (num_rows, dim_size), output is (num_rows, dim_size * repeats)
// Each output element out[i] = input[i / repeats]

struct Params {
    uint numel;    // total output elements
    uint repeats;  // number of repeats per element
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> output;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    uint src_idx = tid.x / params.repeats;
    output[tid.x] = input[src_idx];
}
