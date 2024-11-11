#include <iostream>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <inttypes.h>
#include <assert.h>
#include <string.h>

#define EL_TYPE uint16_t
#define BIG_TILE_SIZE 64
#define MAIN_TILE_WALK 1
// pad LDS row by dword
#define LDS_PAD (4 / sizeof(EL_TYPE))
constexpr uint32_t element_size = sizeof(EL_TYPE);  // in bytes
constexpr uint32_t elements_in_16B = 16 / element_size;

typedef void (*kernel_t)(const void* __restrict a, void* __restrict c, const int N, const int K);

union BLOCK_16B
{
    EL_TYPE e[elements_in_16B];
    __uint128_t ow;
};

template<bool _nt, class _T>
__device__ __forceinline__ _T load(const _T& ref)
{
    if (_nt)
    {
        return __builtin_nontemporal_load(&ref);
    }
    else
    {
        return ref;
    }
}

struct GridLoc
{
    uint32_t i;
    uint32_t j;
    uint32_t k;
};

__device__ __inline__ GridLoc loc_from_idx(uint64_t idx, const int N, const int K)
{
    GridLoc loc;
    loc.i = idx / (uint64_t)(N * K);
    loc.j = (idx / K) % N;
    loc.k = idx % K;
    return loc;
}

template<bool _transpose>
__device__ __inline__ uint64_t calc_offset(GridLoc loc, const int N, const int K)
{
    uint64_t offs = (_transpose ? ((uint64_t)loc.k * N + loc.j) : (uint64_t)(loc.j * K + loc.k)) + (uint64_t)loc.i * N * K;
    return offs;
}

template<class _T, int _vec, bool _transpose>
__global__ void copy_elwise_kernel(const void* __restrict a, void* __restrict c, const int N, const int K)
{
    static_assert(!_transpose || (_vec == 1));

    uint64_t idx = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * _vec;
    GridLoc loc = loc_from_idx(idx, N, K);
    uint64_t offset_a = calc_offset<_transpose>(loc, N, K);
    uint64_t offset_c = calc_offset<false>(loc, N, K);

    const _T* pa = (const _T*)a;
    _T* pc = (_T*)c;
    #pragma unroll
    for (uint32_t v = 0; v < _vec; v++)
    {
        pc[offset_c++] = pa[offset_a++];
    }
}

template<class _T, int _TILE, bool _transpose>
__global__ void copy_tile_kernel(const void* __restrict a, void* __restrict c, const int N, const int K)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Reindex with tiles
    uint64_t tile_idx = idx / (_TILE * _TILE);
    // Walk tiles horizontally
    uint32_t n_tiles = N / _TILE;
    uint32_t k_tiles = K / _TILE;
    uint32_t ti = tile_idx / (k_tiles * n_tiles);
    uint32_t tj = (tile_idx / k_tiles) % n_tiles;
    uint32_t tk = tile_idx % k_tiles;
    uint32_t oj = (idx / _TILE) % _TILE;
    uint32_t ok = idx % _TILE;
    GridLoc loc;
    loc.i = ti;
    loc.j = tj * _TILE + oj;
    loc.k = tk * _TILE + ok;
    uint64_t offset_a = calc_offset<_transpose>(loc, N, K);
    uint64_t offset_c = calc_offset<false>(loc, N, K);

    const _T* pa = (const _T*)a;
    _T* pc = (_T*)c;
    pc[offset_c] = pa[offset_a];
}

template<class _T, int _WG>
__global__ void transpose_tile_big_kernel(const void* __restrict a, void* __restrict c, const int N, const int K)
{
    // Round up processing to next full tile
    const uint32_t n_tiles = (N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
    const uint32_t k_tiles = (K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
    const uint32_t nk_tiles = n_tiles * k_tiles;
    const uint32_t m = blockIdx.x / nk_tiles;
    const uint64_t stride_n = N * sizeof(_T);
    const uint64_t stride_k = K * sizeof(_T);
    const uint64_t stride_nk = N * K * sizeof(_T);

    // Walk destination tiles continuously for cache coherency
    constexpr uint32_t XCD = 8;
    constexpr uint32_t SEQ = 8;
    constexpr uint32_t sblk = XCD * SEQ;
    const uint32_t max_swizzle = (nk_tiles / sblk) * sblk;
    uint32_t tIdx = blockIdx.x % nk_tiles;
    tIdx = tIdx > max_swizzle ? tIdx :
        (tIdx / sblk) * sblk + (tIdx % sblk) / SEQ + (tIdx % SEQ) * XCD;
    uint32_t ti = tIdx / k_tiles;
    uint32_t tj = tIdx % k_tiles;

    __shared__ _T sa[BIG_TILE_SIZE][BIG_TILE_SIZE + LDS_PAD];

    // Detect partial tiles
    uint32_t max_part_n = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE) != 0) ? (N % BIG_TILE_SIZE) : BIG_TILE_SIZE;
    uint32_t max_part_k = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE) != 0) ? (K % BIG_TILE_SIZE) : BIG_TILE_SIZE;

    if (max_part_n == BIG_TILE_SIZE && max_part_k == BIG_TILE_SIZE)
    {
        // Copy full tile with large loads
        constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
        constexpr uint32_t vmem_per_row = row_bytes / sizeof(__uint128_t);
        constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
        constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
        // Make sure WG isn't too large
        static_assert(vmem_per_thread >= 1);

        const uint8_t* pat = (const uint8_t*)a + tj * BIG_TILE_SIZE * stride_n + ti * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            uint64_t offset = row * stride_n + col * sizeof(__uint128_t);
            const __uint128_t* pfa = (const __uint128_t*)(pat + offset);
            BLOCK_16B d;
            d.ow = *pfa;
            #pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
                sa[row][col * elements_in_16B + i] = d.e[i];
            }
        }
        __syncthreads();

        const uint8_t* pc = (const uint8_t*)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
            BLOCK_16B d;
            // Transpose tile on read from LDS
            #pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
                d.e[i] = sa[col * elements_in_16B + i][row];
            }
            __uint128_t* pfc = (__uint128_t*)(pc + offset);
            *pfc = d.ow;
        }
    }
    else
    {
        // Copy partial tiles with element accesses
        constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
        constexpr uint32_t vmem_per_row = BIG_TILE_SIZE;
        constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
        constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
        // Make sure WG isn't too large
        static_assert(vmem_per_thread >= 1);

        const uint8_t* pat = (const uint8_t*)a + tj * BIG_TILE_SIZE * stride_n + ti * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            uint64_t offset = (col < max_part_n && row < max_part_k) ? row * stride_n + col * 2 : 0;
            const uint16_t* pfa = (const uint16_t*)(pat + offset);
            sa[row][col] = *pfa;
        }
        __syncthreads();

        const uint8_t* pc = (const uint8_t*)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            if (col < max_part_k && row < max_part_n)
            {
                uint64_t offset = row * stride_k + col * 2;
                uint16_t* pfc = (uint16_t*)(pc + offset);
                *pfc = sa[col][row];
            }
        }
    }
}
