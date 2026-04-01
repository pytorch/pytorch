// Batched matrix multiplication with optional transposes and output scaling:
//   C[b,M,N] = scale * op(A) * op(B)
// where op(X) = X or X^T depending on transpose flags.
//
// When transpose_a=0: A is [b, M, K]     (contiguous)
// When transpose_a=1: A is [b, K, M]     (contiguous, treated as transposed)
// When transpose_b=0: B is [b, K, N]     (contiguous)
// When transpose_b=1: B is [b, N, K]     (contiguous, treated as transposed)
// scale: multiply output by this constant (use 1.0 to skip; fused with matmul at no extra cost)

struct Params {
    uint batch;
    uint M;
    uint N;
    uint K;
    uint transpose_a;
    uint transpose_b;
    float scale;
};

[[vk::binding(0)]] StructuredBuffer<float> A;
[[vk::binding(1)]] StructuredBuffer<float> B;
[[vk::binding(2)]] RWStructuredBuffer<float> C;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint row = tid.x;
    uint col = tid.y;
    uint b = tid.z;

    if (b >= params.batch || row >= params.M || col >= params.N) return;

    uint c_base = b * params.M * params.N;

    float sum = 0.0f;
    if (params.transpose_a != 0 && params.transpose_b != 0) {
        // A^T * B^T: A is [b,K,M], B is [b,N,K]
        uint a_base = b * params.K * params.M;
        uint b_base = b * params.N * params.K;
        for (uint k = 0; k < params.K; k++) {
            sum += A[a_base + k * params.M + row] * B[b_base + col * params.K + k];
        }
    } else if (params.transpose_a != 0) {
        // A^T * B: A is [b,K,M], B is [b,K,N]
        uint a_base = b * params.K * params.M;
        uint b_base = b * params.K * params.N;
        for (uint k = 0; k < params.K; k++) {
            sum += A[a_base + k * params.M + row] * B[b_base + k * params.N + col];
        }
    } else if (params.transpose_b != 0) {
        // A * B^T: A is [b,M,K], B is [b,N,K]
        uint a_base = b * params.M * params.K;
        uint b_base = b * params.N * params.K;
        for (uint k = 0; k < params.K; k++) {
            sum += A[a_base + row * params.K + k] * B[b_base + col * params.K + k];
        }
    } else {
        // A * B: A is [b,M,K], B is [b,K,N]
        uint a_base = b * params.M * params.K;
        uint b_base = b * params.K * params.N;
        for (uint k = 0; k < params.K; k++) {
            sum += A[a_base + row * params.K + k] * B[b_base + k * params.N + col];
        }
    }
    C[c_base + row * params.N + col] = sum * params.scale;
}
