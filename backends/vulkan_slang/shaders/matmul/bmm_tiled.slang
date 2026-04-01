// Tiled batched matrix multiplication with optional transposes and output scaling:
//   C[b,M,N] = scale * op(A) * op(B)
// Uses shared-memory tiling for improved throughput vs naive per-element bmm.slang.
//
// When transpose_a=0: A is [b, M, K]  (contiguous)
// When transpose_a=1: A is [b, K, M]  (contiguous, treated as transposed)
// When transpose_b=0: B is [b, K, N]  (contiguous)
// When transpose_b=1: B is [b, N, K]  (contiguous, treated as transposed)
// scale: multiply output by this constant (fused at no extra cost)

static const uint TILE = 16;

groupshared float sA[TILE][TILE];
groupshared float sB[TILE][TILE];

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
[numthreads(TILE, TILE, 1)]
void computeMain(
    uint3 gid : SV_GroupID,
    uint3 lid : SV_GroupThreadID)
{
    uint b    = gid.z;
    uint row  = gid.x * TILE + lid.x;
    uint col  = gid.y * TILE + lid.y;
    uint lrow = lid.x;
    uint lcol = lid.y;

    if (b >= params.batch) return;

    uint a_base = b * params.M * params.K;
    uint b_base_stride = b * params.K * params.N;

    float sum = 0.0f;
    uint num_tiles = (params.K + TILE - 1) / TILE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tile of A
        uint ak = t * TILE + lcol;
        if (params.transpose_a != 0) {
            // A is [b, K, M], we want A^T[row, k] = A[k, row]
            if (row < params.M && ak < params.K)
                sA[lrow][lcol] = A[b * params.K * params.M + ak * params.M + row];
            else
                sA[lrow][lcol] = 0.0f;
        } else {
            // A is [b, M, K]
            if (row < params.M && ak < params.K)
                sA[lrow][lcol] = A[a_base + row * params.K + ak];
            else
                sA[lrow][lcol] = 0.0f;
        }

        // Load tile of B
        uint bk = t * TILE + lrow;
        if (params.transpose_b != 0) {
            // B is [b, N, K], we want B^T[k, col] = B[col, k]
            if (bk < params.K && col < params.N)
                sB[lrow][lcol] = B[b * params.N * params.K + col * params.K + bk];
            else
                sB[lrow][lcol] = 0.0f;
        } else {
            // B is [b, K, N]
            if (bk < params.K && col < params.N)
                sB[lrow][lcol] = B[b_base_stride + bk * params.N + col];
            else
                sB[lrow][lcol] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        for (uint k = 0; k < TILE; k++) {
            sum += sA[lrow][k] * sB[k][lcol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < params.M && col < params.N) {
        C[b * params.M * params.N + row * params.N + col] = sum * params.scale;
    }
}
