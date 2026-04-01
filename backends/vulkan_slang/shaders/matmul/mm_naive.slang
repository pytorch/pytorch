// Naive matrix multiplication: C[M,N] = A[M,K] * B[K,N]

struct Params {
    uint M;
    uint N;
    uint K;
};

[[vk::binding(0)]] StructuredBuffer<float> A;
[[vk::binding(1)]] StructuredBuffer<float> B;
[[vk::binding(2)]] RWStructuredBuffer<float> C;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    uint row = tid.x;
    uint col = tid.y;

    if (row >= params.M || col >= params.N) return;

    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.K + k] * B[k * params.N + col];
    }
    C[row * params.N + col] = sum;
}
