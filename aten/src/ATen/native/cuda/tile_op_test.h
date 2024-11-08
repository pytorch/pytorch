#include <iostream>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#define _CHECK(condition)                                                                   \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            exit(1);                                                                  \
        }                                                                                   \
    }

class Timer
{
public:
    void Set(uint32_t delay_ms)
    {
        timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        et_ = t.tv_sec * 1000000000ULL + t.tv_nsec + delay_ms * 1000000ULL;
    }

    bool IsExpired()
    {
        timespec t;
        clock_gettime(CLOCK_MONOTONIC, &t);
        uint64_t ct = t.tv_sec * 1000000000ULL + t.tv_nsec;
        return ct > et_;
    }

private:
    uint64_t et_;
};

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

uint32_t test_iters = 100;
uint32_t prewarm_ms = 50;
bool fill_zero = false;
bool nt_loads = false;
bool report_metric = true;

const uint64_t mall_size = 256 * 1024 * 1024;
uint32_t cu_cap = 0xffffffff;

hipStream_t stream = 0;
uint64_t data_size = 0;
// void* buf_a;
// void* buf_c;

// template<typename T>
// T* transpose_cpu_nk(T* src_gpu, int M, int N, int K) {
//     T *src = (T *)malloc((uint64_t)M * N * K * sizeof(T));
//     T *dst = (T *)malloc((uint64_t)M * N * K * sizeof(T));
//     _CHECK(hipMemcpy(src, src_gpu, (uint64_t)M * N * K * sizeof(T), hipMemcpyDeviceToHost));
//     for (int s = 0; s < M; ++s) {
//         for (int i = 0; i < N; ++i) {
//             for (int j = 0; j < K; ++j) {
//                 dst[(uint64_t)j * N + i + (uint64_t)s * N * K] = src[(uint64_t)i * K + j + (uint64_t)s * N * K];
//             }
//         }
//     }
//     free(src);
//     return dst;
// }

// template<typename T>
// void compare_nk(T* ref_cpu, T *val_gpu, int M, int N, int K) {
//     T *temp_cpu = (T *)malloc((uint64_t)M * N * K * sizeof(T));
//     _CHECK(hipMemcpy(temp_cpu, val_gpu, (uint64_t)M * N * K * sizeof(T), hipMemcpyDeviceToHost));
//     int print_cnt = 0;
//     for (int s = 0; s < M; ++s) {
//         for (int i = 0; i < K; ++i) {
//             for (int j = 0; j < N; ++j) {
//                 if(temp_cpu[(uint64_t)i * N + j + (uint64_t)s * N * K] != ref_cpu[(uint64_t)i * N + j + (uint64_t)s * N * K]) {
//                     printf("error: %d %d %d ref %d val %d\n", s, j, i, ref_cpu[(uint64_t)i * N + (uint64_t)j + s * N * K], temp_cpu[(uint64_t)i * M + j + (uint64_t)s * N * K]);
//                     print_cnt++;
//                     if(print_cnt > 10){
//                         free(temp_cpu);
//                         return;
//                     }
//                 }
//             }
//         }
//     }
//     free(temp_cpu);
// }

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

void adv_ptr(void** p, void* base, size_t size, size_t inc)
{
    uint8_t* pp = (uint8_t*)p + inc;
    if (pp > (uint8_t*)base + size) {
        *p = base;
    }
    else {
        *p = (void*)pp;
    }
}

void RunTest(kernel_t kernel, const std::string& test_name, 
             int wg_count, int wg_size, int vec, const int M, const int N, const int K, void *buf_a, void *buf_c)
{
    const dim3 grid_dim(wg_count, 1, 1);
    const dim3 block_dim(wg_size, 1, 1);
    const uint64_t tsize = (uint64_t)M * N * N * sizeof(EL_TYPE);

    hipEvent_t start;
    hipEvent_t stop;
    _CHECK(hipEventCreate(&start));
    _CHECK(hipEventCreate(&stop));

    kernel<<<grid_dim, block_dim, 0, stream>>>(buf_a, buf_c, N, K);
    // if (ref_cpu != nullptr)
    // {
    //     compare_nk((EL_TYPE *)ref_cpu, (EL_TYPE *)buf_c, M, N, K);
    // }

    void* pa = buf_a;
    void* pc = buf_c;

    kernel<<<grid_dim, block_dim, 0, stream>>>(pa, pc, N, K);
    _CHECK(hipStreamSynchronize(stream));

    adv_ptr(&pa, buf_a, data_size, tsize);
    adv_ptr(&pc, buf_c, data_size, tsize);
    
    Timer tmr;
    tmr.Set(prewarm_ms);
    while (!tmr.IsExpired())
    {
        kernel<<<grid_dim, block_dim, 0, stream>>>(pa, pc, N, K);
        adv_ptr(&pa, buf_a, data_size, tsize);
        adv_ptr(&pc, buf_c, data_size, tsize);
        _CHECK(hipStreamSynchronize(stream));
    }

    double min_bw = __FLT_MAX__;
    double max_bw = 0.0f;
    double avg_bw = 0.0f;

    for (uint32_t i = 0; i < test_iters; i++)
    {
        hipExtLaunchKernelGGL(kernel, grid_dim, block_dim, 0, stream, start, stop, 0, pa, pc, N, K);
        adv_ptr(&pa, buf_a, data_size, tsize);
        adv_ptr(&pc, buf_c, data_size, tsize);

        _CHECK(hipEventSynchronize(start));
        _CHECK(hipEventSynchronize(stop));
        float t_ms;
        _CHECK(hipEventElapsedTime(&t_ms, start, stop));
        const uint64_t elements = (uint64_t)M * N * K * 2;
        double bw = (double)elements * element_size / (t_ms  / 1000.0) / 1000000000;
        if (!report_metric)
        {
            bw /= 1.024 * 1.024 * 1.024;
        }

        min_bw = std::min(min_bw, bw);
        max_bw = std::max(max_bw, bw);
        avg_bw += bw;
    }
    _CHECK(hipStreamSynchronize(stream));

    avg_bw /= test_iters;

    printf("%s,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f\n",
        test_name.c_str(), wg_size, vec, M, N, K, min_bw, max_bw, avg_bw);

    _CHECK(hipEventDestroy(start));
    _CHECK(hipEventDestroy(stop));
}

// Transposed layouts (except big tiles) won't use NT
void RunTests(const int M, const int N, const int K, void *buf_a, void *buf_c)
{
    uint64_t elements = (uint64_t)M * N * K;
    // EL_TYPE* ref_cpu = transpose_cpu_nk<EL_TYPE>((EL_TYPE *)buf_a, M, K, N);

    for (int wg = 64; wg <= 1024 && wg <= elements; wg *= 2) {
        int elwise_wg = elements / wg;
        RunTest(copy_elwise_kernel<EL_TYPE, 1, false>, 
                "Copy elementwise,N", elwise_wg, wg, 1, M, N, K, buf_a, buf_c);
        if (elements % 8 == 0) {
            RunTest(copy_elwise_kernel<EL_TYPE, 8, false>, 
                "Copy elementwise,N", elwise_wg / 8, wg, 8, M, N, K, buf_a, buf_c);
        }
        RunTest(copy_elwise_kernel<EL_TYPE, 1, true>, 
                "Copy elementwise,T", elwise_wg, wg, 1, M, N, K, buf_a, buf_c);

        if (N % 8 == 0 && K % 8 == 0) {
            RunTest(copy_tile_kernel<EL_TYPE,  8, true>, 
                    "Copy tile 8x8,T", elwise_wg, wg, 1, M, N, K, buf_a, buf_c);
        }
    }

    int big_tile_wg = M * ((N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE) * ((K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE);
    RunTest(transpose_tile_big_kernel<EL_TYPE, 64>, 
            "Custom transpose LDS tile 64x64,T", big_tile_wg, 64, 1, M, N, K, buf_a, buf_c);
    RunTest(transpose_tile_big_kernel<EL_TYPE, 128>, 
            "Custom transpose LDS tile 64x64,T", big_tile_wg, 128, 1, M, N, K, buf_a, buf_c);
    RunTest(transpose_tile_big_kernel<EL_TYPE, 256>, 
            "Custom transpose LDS tile 64x64,T", big_tile_wg, 256, 1, M, N, K, buf_a, buf_c);
    RunTest(transpose_tile_big_kernel<EL_TYPE, 512>, 
            "Custom transpose LDS tile 64x64,T", big_tile_wg, 512, 1, M, N, K, buf_a, buf_c);

    // free(ref_cpu);
}

// void FillMemory(EL_TYPE* p, size_t size)
// {
//     EL_TYPE* r{};
// #if 0
//     uint64_t v = 134775813;
//     _CHECK(hipHostMalloc(&r, size));
//     for (size_t i = 0; i < size / sizeof(EL_TYPE); i++)
//     {
//         v = v * 6364136223846793005 + 1442695040888963407;
//         r[i] = v & 0xffff;
//     }
//     _CHECK(hipMemcpy(p, r, size, hipMemcpyDefault));
// #else
//     const size_t fitems = 265 * 1024;
//     const size_t fsize = fitems * sizeof(EL_TYPE);
//     _CHECK(hipHostMalloc(&r, fsize));
//     for (size_t i = 0; i < fitems; i++)
//     {
//         r[i] = fill_zero ? 0 : rand();
//     }

//     while (size > 0)
//     {
//         _CHECK(hipMemcpy(p, r, std::min(fsize, size), hipMemcpyDefault));
//         p += fitems;
//         size -= std::min(fsize, size);
//     }
// #endif
//     _CHECK(hipHostFree(r));
// }

// void ParseParams(char** argv)
// {
//     int i = 1;
//     while (argv[i] != NULL)
//     {
//         if (strcmp(argv[i], "-i") == 0)
//         {
//             if (argv[i + 1] == NULL)
//             {
//                 printf("Missing test iterations\n");
//                 exit(1);
//             }
//             test_iters = atoi(argv[i + 1]);
//             i += 2;
//         }
//         else if (strcmp(argv[i], "-p") == 0)
//         {
//             if (argv[i + 1] == NULL)
//             {
//                 printf("Missing prewarm duraion in ms\n");
//                 exit(1);
//             }
//             prewarm_ms = atoi(argv[i + 1]);
//             i += 2;
//         }
//         else if (strcmp(argv[i], "-nt") == 0)
//         {
//             if (argv[i + 1] == NULL)
//             {
//                 printf("Missing non-temporal load flag value\n");
//                 exit(1);
//             }
//             nt_loads = atoi(argv[i + 1]) != 0;
//             i += 2;
//         }
//         else if (strcmp(argv[i], "-mu") == 0)
//         {
//             if (argv[i + 1] == NULL)
//             {
//                 printf("Missing metric units flag value\n");
//                 exit(1);
//             }
//             report_metric = atoi(argv[i + 1]) != 0;
//             i += 2;
//         }
//         else if (strcmp(argv[i], "-z") == 0)
//         {
//             fill_zero = true;
//             i += 1;
//         }
//         else if (strcmp(argv[i], "-cu") == 0)
//         {
//             if (argv[i + 1] == NULL)
//             {
//                 printf("Missing max CU value\n");
//                 exit(1);
//             }
//             cu_cap = atoi(argv[i + 1]);
//             i += 2;
//         }
//         else
//         {
//             printf("Unknown parameter %s\n", argv[i]);
//             exit(1);
//         }
//     }
// }

// int main(int argc, char **argv)
// {
//     if (argc > 1)
//     {
//         ParseParams(argv);
//     }

//     static const uint32_t max_cu = 1024;
//     if (cu_cap > max_cu)
//     {
//         _CHECK(hipStreamCreate(&stream));
//     }
//     else
//     {
//         static const uint32_t mask_count = max_cu / 32;
//         uint32_t cu_mask[mask_count] = {};
//         _CHECK(hipExtStreamGetCUMask(0, mask_count, cu_mask));

//         uint32_t remaining_cu = cu_cap;
//         for (uint32_t cu = 0; cu < max_cu; cu++)
//         {
//             uint32_t mask_idx = cu / 32;
//             uint32_t mask = 1 << (cu % 32);
//             if (remaining_cu == 0)
//             {
//                 cu_mask[mask_idx] &= ~mask;
//             }
//             else
//             {
//                 if (cu_mask[mask_idx] & mask)
//                 {
//                     remaining_cu--;
//                 }
//             }
//         }

//         if (remaining_cu > 0)
//         {
//             printf("%d/%d CUs remain unallocated\n", remaining_cu, cu_cap);
//         }

//         _CHECK(hipExtStreamCreateWithCUMask(&stream, mask_count, cu_mask));
//     }

//     int dev;
//     _CHECK(hipGetDevice(&dev));
//     uint32_t M,N,K;
//     uint32_t shapes[][3] = {
//         {1,1792,156},
//         // {1,1792,312},
//         // {1,1792,512},
//         // {1,1792,2048},
//         // {1,1792,3072},
//         // {1,1792,5972},
//         // {1,1792,56672},
//         // {1,2048,128},
//         // {1,2048,256},
//         // {1,2048,280},
//         // {1,2048,300},
//         // {1,2048,360},
//         // {1,2048,384},
//         // {1,2048,1024},
//         // {1,2048,1320},
//         // {1,2048,1888},
//         // {1,2048,1968},
//         // {1,2048,1976},
//         // {1,2048,1980},
//         // {1,2048,1984},
//         // {1,2048,1992},
//         // {1,2048,2048},
//         // {1,2048,2304},
//         // {1,2048,4800},
//         // {1,2048,6016},
//         // {1,2048,8064},
//         // {1,2048,10112},
//         // {1,2048,12160},
//         // {1,2048,14208},
//         // {1,2048,16256},
//         // {1,2048,18304},
//         // {1,2048,20352},
//         // {1,2048,22400},
//         // {1,2048,24448},
//         // {1,2048,26496},
//         // {1,3072,24},
//         // {1,3072,128},
//         // {1,3072,160},
//         // {1,3072,256},
//         // {1,3072,316},
//         // {1,3072,320},
//         // {1,3072,440},
//         // {1,3072,512},
//         // {1,3072,768},
//         // {1,3072,888},
//         // {1,3072,1024},
//         // {1,3072,1536},
//         // {1,3072,2048},
//         // {1,3072,3840},
//         // {1,3072,4200},
//         // {1,3072,4480},
//         // {1,3072,5120},
//         // {1,3072,48760},
//         // {1,323584,256},
//         // {200,2048,384},
//         // {1792,32,156},
//         // {1792,48,156},
//         // {1792,61,156},
//         // {1792,109,156},
//         // {1792,156,24},
//         // {1792,156,61},
//         // {1792,156,109},
//         // {1792,156,117},
//         // {1792,156,248},
//         // {1792,156,1771},
//         // {1792,1771,156},
//         // {2048,128,192},
//         // {2048,192,256},
//         // {2048,192,257},
//         // {2048,192,384},
//         // {2048,192,641},
//         // {2048,192,6514},
//         // {2048,200,384},
//         // {2048,256,192},
//         // {2048,257,192},
//         // {2048,384,200},
//         // {3072,32,160},
//         // {3072,48,160},
//         // {3072,57,160},
//         // {3072,105,160},
//         // {3072,121,160},
//         // {3072,160,32},
//         // {3072,160,57},
//         // {3072,160,105},
//         // {3072,160,121},
//         // {3072,160,208},
//         // {3072,160,1219},
//         // {3072,1219,160},
//     };

//     uint64_t max_elements = 0;
//     for(auto shape : shapes)
//     {
//         M = shape[0];
//         N = shape[1];
//         K = shape[2];
//         uint64_t elements = (uint64_t)M * N * K;
//         max_elements = std::max(max_elements, elements);
//     }
//     data_size = std::max(max_elements * element_size, mall_size * 2);
//     printf("Alloc size %.3f MB\n", data_size / 1000000.0f);

//     _CHECK(hipMalloc(&buf_a, data_size));
//     FillMemory((EL_TYPE*)buf_a, data_size);
//     _CHECK(hipMalloc(&buf_c, data_size));
//     FillMemory((EL_TYPE*)buf_c, data_size);

//     for(auto shape : shapes)
//     {
//         M = shape[0];
//         N = shape[1];
//         K = shape[2];
//         RunTests(M, N, K);

//     }

//     _CHECK(hipFree(buf_a));
//     _CHECK(hipFree(buf_c));
//     if (stream != 0)
//     {
//         _CHECK(hipStreamDestroy(stream));
//     }
//     return 0;
// }

