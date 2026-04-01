// Tiled matrix multiplication using shared memory.
//   C[M,N] = op(A) * op(B)
// When transpose_a=0: A is [M,K]  (contiguous), op(A) = A
// When transpose_a=1: A is [K,M]  (contiguous), op(A) = A^T
// When transpose_b=0: B is [K,N]  (contiguous), op(B) = B
// When transpose_b=1: B is [N,K]  (contiguous), op(B) = B^T
//
// Each thread block computes a TILE_SIZE x TILE_SIZE tile of C.

static const uint TILE_SIZE = 16;

groupshared float sA[TILE_SIZE][TILE_SIZE];
groupshared float sB[TILE_SIZE][TILE_SIZE];

struct Params {
    uint M;
    uint N;
    uint K;
    uint transpose_a;
    uint transpose_b;
};

[[vk::binding(0)]] StructuredBuffer<float> A;
[[vk::binding(1)]] StructuredBuffer<float> B;
[[vk::binding(2)]] RWStructuredBuffer<float> C;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void computeMain(
    uint3 tid : SV_DispatchThreadID,
    uint3 lid : SV_GroupThreadID)
{
    uint row = tid.x;
    uint col = tid.y;
    uint lrow = lid.x;
    uint lcol = lid.y;

    float sum = 0.0f;

    uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tile of A into shared memory
        uint ak = t * TILE_SIZE + lcol;
        if (params.transpose_a != 0) {
            // A is [K, M], we want A^T[row, k] = A[k, row]
            if (row < params.M && ak < params.K)
                sA[lrow][lcol] = A[ak * params.M + row];
            else
                sA[lrow][lcol] = 0.0f;
        } else {
            // A is [M, K]
            if (row < params.M && ak < params.K)
                sA[lrow][lcol] = A[row * params.K + ak];
            else
                sA[lrow][lcol] = 0.0f;
        }

        // Load tile of B into shared memory
        uint bk = t * TILE_SIZE + lrow;
        if (params.transpose_b != 0) {
            // B is [N, K], we want B^T[k, col] = B[col, k]
            if (bk < params.K && col < params.N)
                sB[lrow][lcol] = B[col * params.K + bk];
            else
                sB[lrow][lcol] = 0.0f;
        } else {
            // B is [K, N]
            if (bk < params.K && col < params.N)
                sB[lrow][lcol] = B[bk * params.N + col];
            else
                sB[lrow][lcol] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // Multiply tiles
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += sA[lrow][k] * sB[k][lcol];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}
