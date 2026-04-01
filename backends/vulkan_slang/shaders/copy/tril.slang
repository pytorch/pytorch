// Lower triangular (batched): out = input where col <= row + diagonal, else 0
// Works on flat buffer with repeating (rows x cols) matrices.

struct Params {
    uint rows;
    uint cols;
    int diagonal;
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    uint matrix_size = params.rows * params.cols;
    uint local = tid.x % matrix_size;
    uint row = local / params.cols;
    uint col = local % params.cols;

    if ((int)col <= (int)row + params.diagonal) {
        out[tid.x] = input[tid.x];
    } else {
        out[tid.x] = 0.0f;
    }
}
