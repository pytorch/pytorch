#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

Tensor cudnn_mha(const Tensor& self, const Tensor& qkv_weight, const Tensor& out_weight, const Tensor& qkv_bias, const Tensor& out_bias, const long inputSize, const long seqLength, const long batchSize, const long outputSize, const long head_size, const long num_heads) {
    AT_ERROR("cudnn_mha: ATen not compiled with cuDNN support");
  }


}} // namespace at::native

#else // AT_CUDNN_ENABLED
#include <ATen/native/cudnn/MHA.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#include <ATen/cuda/Exceptions.h>
#include <cudnn_frontend.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <iostream>

namespace at { namespace native {

namespace {



#if (CUDNN_VERSION >= 8900)

enum class MHA_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2,
    SBH_INTERLEAVED = 3
};

struct MHAParams {
  c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  MHA_Layout layout;
  int64_t b;
  int64_t h;
  int64_t s_q;
  int64_t s_kv;
  int64_t d;
  bool is_training;

};

void setMHAParams(MHAParams& params,
		  int64_t b,
                  int64_t h,
                  int64_t s_q,
                  int64_t s_kv,
                  int64_t d,
		  MHA_Layout layout,
                  bool is_training,
                  cudnnDataType_t dataType) {
  memset(&params, 0, sizeof(params));
  params.device_id = at::cuda::current_device();
  params.dataType = dataType;
  params.layout = layout; // TODO(eqy): support other layouts?
  params.b = b;
  params.h = h;
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.is_training = is_training;
}

template <typename T, typename KeyType>
struct BenchmarkCache {
std::mutex mutex;
std::unordered_map<KeyType, cudnn_frontend::ExecutionPlan, ParamsHash<KeyType>, ParamsEqual<KeyType>> engine_cache;

cudnn_frontend::ExecutionPlan* find(const KeyType& key) {
  std::lock_guard<std::mutex> guard(mutex);
  auto it = engine_cache.find(key);
  if (it == engine_cache.end()) {
    return nullptr;
  }
  return &(it->second);
}

void update(const KeyType& key, T& results) {
  std::lock_guard<std::mutex> guard(mutex);
  engine_cache.erase(key);
  engine_cache.emplace(key, std::move(results));
}

};

BenchmarkCache<cudnn_frontend::ExecutionPlan, MHAParams> benchmark_cache;

// typedef __half half1;

#include <cudnn_frontend.h>
// #include "error_util.h"

#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

enum class MHA_Matrix {
    Q_Matrix           = 0, // queries
    K_Matrix           = 1, // keys
    K_Matrix_Transpose = 2, // keys tranposed
    V_Matrix           = 3, // values
    V_Matrix_Transpose = 4, // values transposed
    S_Matrix           = 5, // output of GEMM1
    O_Matrix           = 6, // final output
};

enum class MHA_Bias_Type {
    NO_BIAS = 0,
    PRE_SCALE_BIAS = 1,
    POST_SCALE_BIAS = 2
};


// Host functions for converting between FP32 and FP16 formats
// Paulius Micikevicius (pauliusm@nvidia.com)

__half cpu_float2half_rn(float f)
{
    void* f_ptr = &f;
    unsigned x = *((int*)f_ptr);
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    __half_raw hr;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        hr.x = 0x7fffU;
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<__half*>(hr_ptr);
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        hr.x = static_cast<unsigned short> (sign | 0x7c00U);
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<__half*>(hr_ptr);
    }
    if (u < 0x33000001) {
        hr.x = static_cast<unsigned short> (sign | 0x0000U);
        // Add an indirection to get around type aliasing check
        void* hr_ptr = &hr;
        return *reinterpret_cast<__half*>(hr_ptr);
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    hr.x = static_cast<unsigned short>((sign | (exponent << 10) | mantissa));

    // Add an indirection to get around type aliasing check
    void* hr_ptr = &hr;
    return *reinterpret_cast<__half*>(hr_ptr);
}

// Used for MHA
void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, int64_t* strideA, MHA_Layout layout, MHA_Matrix matrix) {
    CUDNN_FRONTEND_UNUSED(b);
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix)
    {
        case MHA_Matrix::Q_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * 3 * h * d;

            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d * b;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = 3 * d;
                strideA[batch_dim_idx] = 3 * h * d;
            } else {
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * h * d;
            }
            break;
        case MHA_Matrix::K_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 2 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d * b;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = 3 * d;
                strideA[batch_dim_idx] = 3 * h * d;
            } else {
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::K_Matrix_Transpose:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d * b;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = 3 * d;
                strideA[batch_dim_idx] = 3 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 2* h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d * b;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = 3 * d;
                strideA[batch_dim_idx] = 3 * h * d;
            } else {
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix_Transpose:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2* h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else if (layout == MHA_Layout::SBH_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d * b;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = 3 * d;
                strideA[batch_dim_idx] = 3 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::S_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_q * s_kv;
            strideA[batch_dim_idx] = h * s_q * s_kv;
            break;
        case MHA_Matrix::O_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = h * d;
            strideA[head_dim_idx] = d;
            strideA[batch_dim_idx] = s_q * h * d;
            break;
    }
}

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type, int64_t id, int64_t const * dim,
                                int64_t const * stride, bool is_virtual, bool is_value) {
    int nbDims = 4;
    auto tensor_created = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, dim)
            .setStride(nbDims, stride)
            .setId(id)
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(type)
            .setVirtual(is_virtual)
            .setByValue(is_value)
            .build();
    return tensor_created;
};

static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type, cudnnPointwiseMode_t mode) {
    auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
            .setMode(mode)
            .setComputeType(type)
            .build();

    return pw_desc_created;
}

static cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &yDesc,
                                                    cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    return pw_op_created;
}

static cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &bDesc,
                                                    cudnn_frontend::Tensor const &yDesc, cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setbDesc(bDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    return pw_op_created;
}

static cudnn_frontend::Operation ternary_pw_op_create(cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &bDesc, cudnn_frontend::Tensor const &tDesc,
                                            cudnn_frontend::Tensor const &yDesc, cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(xDesc)
                        .setbDesc(bDesc)
                        .settDesc(tDesc)
                        .setyDesc(yDesc)
                        .setpwDesc(pwDesc)
                        .build();
    return pw_op_created;
}

static cudnn_frontend::Tensor
createScale(int64_t b,
            int64_t h,
            int64_t s_q,
            int64_t s_kv,
            int64_t d,
            MHA_Layout layout,
            cudnnDataType_t tensorType,
            const cudnn_frontend::Tensor& sTensor,
            std::vector<cudnn_frontend::Operation>& ops) {

    // scale
    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    int64_t s_dim [4] =  {b, h, s_q, s_kv};
    int64_t s_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    auto scaleTensor = tensor_create(tensorType, S_CONST_ID, scale_dim, scale_stride, false, true); // is by value
    auto sScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 2000, s_dim, s_stride, true, false); // is virtual

    // Define the scale descriptor
    auto scaleDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a Scale Node.
    auto scale_op = binary_pw_op_create(sTensor, scaleTensor, sScaleTensor, scaleDesc);

    ops.push_back(std::move(scale_op));
    return sScaleTensor;
}

static cudnn_frontend::Tensor
createQKBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops) {
    // Creates the necessary tensor descriptors
    int64_t q_dim [4] = {b, h, s_q, d};
    int64_t q_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

    int64_t k_dim [4] =  {b, h, d, s_kv};
    int64_t k_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix_Transpose);

    int64_t s_dim [4] = {b, h, s_q, s_kv};
    int64_t s_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::S_Matrix);

    auto qTensor = tensor_create(tensorType, Q_ID, q_dim, q_stride, false, false);
    auto kTransposeTensor = tensor_create(tensorType, K_ID, k_dim, k_stride, false, false); // is virtual
    // first GEMM output
    auto sTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 1, s_dim, s_stride, true, false); // is virtual

    // Define the matmul 1 desc
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();

    // Create a matmul 1 Node
    auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(qTensor)
                            .setbMatDesc(kTransposeTensor)
                            .setcMatDesc(sTensor)
                            .setmatmulDesc(matmul_1_Desc)
                            .build();


    ops.push_back(std::move(matmul_op1));

    return sTensor;
}

static cudnn_frontend::Tensor
createCausalMask(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           cudnn_frontend::Tensor& prevBlockOutputTensor) {

    CUDNN_FRONTEND_UNUSED(d);
    CUDNN_FRONTEND_UNUSED(layout);
    CUDNN_FRONTEND_UNUSED(tensorType);

    cudnn_frontend::throw_if(ops.size() == 0, "Padding Mask constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    // subtraction output
    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t maskVal_dim [4] =  {1, 1, 1, 1};
    int64_t maskVal_stride [4] = {1, 1, 1, 1};

    // mask value to put in the masked pixels
    auto maskValTensor = tensor_create(CUDNN_DATA_FLOAT, MASK_VAL_ID, maskVal_dim, maskVal_stride, false, true); // is by value
    // gen index row output
    auto rowIndexTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 100, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // gen index column output
    auto columnIndexTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 101, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // create causal mask (row >= col)
    auto causalMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN, VIRTUAL_ID + 106, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // output after masking
    auto maskOutputTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 107, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // Define the gen index for row descriptor
    auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_GEN_INDEX)
                            .setAxis(2)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

    // Create a gen index Node.
    auto genIndexRow_op = unary_pw_op_create(prevBlockOutputTensor, rowIndexTensor, genIndexRowDesc);

    // Define the gen index for row descriptor
    auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_GEN_INDEX)
                            .setAxis(3)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();

    // Create a gen index Node.
    auto genIndexColumn_op = unary_pw_op_create(prevBlockOutputTensor, columnIndexTensor, genIndexColumnDesc);

    // Define the greater than equal to comparison descriptor
    auto rowGreaterColDesc = pw_desc_create(CUDNN_DATA_BOOLEAN, CUDNN_POINTWISE_CMP_GE);

    // Create a greater than equal to Node.
    auto rowGreaterCol_op = binary_pw_op_create(rowIndexTensor, columnIndexTensor, causalMaskTensor, rowGreaterColDesc);

    /////////////////// Apply the mask //////////////////////////

    // Define the binary select to perform masking descriptor
    auto maskDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_BINARY_SELECT);

    // Create a binary select Node.
    auto mask_op = ternary_pw_op_create(prevBlockOutputTensor, maskValTensor, causalMaskTensor, maskOutputTensor, maskDesc);

    ops.push_back(std::move(genIndexRow_op));
    ops.push_back(std::move(genIndexColumn_op));
    ops.push_back(std::move(rowGreaterCol_op));
    ops.push_back(std::move(mask_op));

    return maskOutputTensor;
}

static cudnn_frontend::Tensor
createSoftmaxForward(int64_t b,
                     int64_t h,
                     int64_t s_q,
                     int64_t s_kv,
                     bool is_training,
                     std::vector<cudnn_frontend::Operation>& ops,
                     cudnn_frontend::Tensor& sAfterMaskTensor) {

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t afterReduction_dim [4] = {b, h, s_q, 1};
    int64_t afterReduction_stride [4] = {h * s_q, s_q, 1, 1};

    // max (x)
    auto afterMaxReductionTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 150, afterReduction_dim, afterReduction_stride, true, false); // is virtual

    // x - max(x)
    auto afterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 151, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual

    // e^(x - max(x))
    auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 152, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual;

    // sum (e^(x - max(x)))
    auto afterAddReductionTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 153, afterReduction_dim, afterReduction_stride, true, false); // is virtual

    // log (sum (e^(x - max(x))))
    auto afterLogLTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 154, afterReduction_dim, afterReduction_stride, true, false);

    // M + log (sum (e^(x - max(x))))
    auto softmaxStatsTensor = tensor_create(CUDNN_DATA_FLOAT, S_STATS_ID, afterReduction_dim, afterReduction_stride, !is_training, false); // not virtual if training is true, virtual if training is false

    // divide (e/ sum(e))
    auto afterSoftmaxTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, afterBMM1_dim)
            .setStride(4, afterBMM1_stride)
            .setId(VIRTUAL_ID + 156)
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(CUDNN_DATA_FLOAT)
            .setVirtual(true)
            .setByValue(false)
            .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
            .build();

    // Define the reduction descriptor
    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                                .build();

    // Create a reduction max Node.
    auto reductionMax_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(sAfterMaskTensor)
                                .setyDesc(afterMaxReductionTensor)
                                .setreductionDesc(reductionMaxDesc)
                                .build();

    // Define the subtract descriptor
    auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);

    // Create a subtract Node.
    auto subtract_op = binary_pw_op_create(sAfterMaskTensor, afterMaxReductionTensor, afterSubtractionTensor, subtractDesc);

    // Define the exponent descriptor
    auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);

    // Create a exponent Node.
    auto exponent_op = unary_pw_op_create(afterSubtractionTensor, afterExponentTensor, exponentDesc);

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                .build();

    // Create a reduction add Node.
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterExponentTensor)
                                .setyDesc(afterAddReductionTensor)
                                .setreductionDesc(reductionAddDesc)
                                .build();


    // Create log descriptor
    auto logDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_LOG);

    // Create log node
    auto log_op = unary_pw_op_create(afterAddReductionTensor, afterLogLTensor, logDesc);

    // Create add descriptor
    auto addDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_ADD);

    // Create add node
    auto add_op = binary_pw_op_create(afterMaxReductionTensor, afterLogLTensor, softmaxStatsTensor, addDesc);

    // Define the division descriptor
    auto divisionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_DIV);

    // Create a subtract Node.
    auto division_op = binary_pw_op_create(afterExponentTensor, afterAddReductionTensor, afterSoftmaxTensor, divisionDesc);

    ops.push_back(std::move(reductionMax_op));
    ops.push_back(std::move(subtract_op));
    ops.push_back(std::move(exponent_op));
    ops.push_back(std::move(reductionAdd_op));
    ops.push_back(std::move(log_op));
    ops.push_back(std::move(add_op));
    ops.push_back(std::move(division_op));

    return afterSoftmaxTensor;
}

static cudnn_frontend::Tensor
createDropoutForward(int64_t b,
              int64_t h,
              int64_t s_q,
              int64_t s_kv,
              int64_t d,
              double probability,
              cudnnDataType_t tensorType,
              std::vector<cudnn_frontend::Operation>& ops,
              cudnn_frontend::Tensor& afterSoftmaxTensor) {

    CUDNN_FRONTEND_UNUSED(d);

    cudnn_frontend::throw_if(ops.size() == 0, "Dropout DAG constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};

    auto dropoutSeed = tensor_create(CUDNN_DATA_INT64, D_SEED_ID, scale_dim, scale_stride, false, false); // not virtual
    auto dropoutOffset = tensor_create(CUDNN_DATA_INT64, D_OFFSET_ID, scale_dim, scale_stride, false, false); // not virtual

    // mask for the dropout
    auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT, VIRTUAL_ID + 200, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    // after dropout tensor
    auto afterDropoutTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, afterBMM1_dim)
            .setStride(4, afterBMM1_stride)
            .setId(VIRTUAL_ID + 201)
            .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            .setDataType(tensorType)
            .setVirtual(true)
            .setByValue(false)
            .setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16)
            .build();
    // scale after dropout
    auto scaleDropoutTensor = tensor_create(tensorType, D_CONST_ID, scale_dim, scale_stride, false, true); // is by value
    // after Scale
    auto afterScaleTensor = tensor_create(tensorType, VIRTUAL_ID + 202, afterBMM1_dim, afterBMM1_stride, true, false); // is virtual


    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                                .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                .setBernoulliDistProbability(1.0 - probability)
                                .build();

    // Create a rng Node.
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                                .setyDesc(dropoutMaskTensor)
                                .setSeedDesc(dropoutSeed)
                                .setOffsetDesc(dropoutOffset)
                                .setRngDesc(rngDesc)
                                .build();


    // Define the multiply mask descriptor
    auto maskMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply mask Node.
    auto maskMul_op = binary_pw_op_create(afterSoftmaxTensor, dropoutMaskTensor, afterDropoutTensor, maskMulDesc);

    // Define the multiply scale descriptor
    auto scaleMulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);

    // Create a multiply scale Node.
    auto scaleMul_op = binary_pw_op_create(afterDropoutTensor, scaleDropoutTensor, afterScaleTensor, scaleMulDesc);

    ops.push_back(std::move(rng_op));
    ops.push_back(std::move(maskMul_op));
    ops.push_back(std::move(scaleMul_op));

    return afterScaleTensor;
}

static void
createSVBMM(int64_t b,
           int64_t h,
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops,
           cudnn_frontend::Tensor const &afterScaleDropoutTensor) {

    cudnn_frontend::throw_if(ops.size() == 0, "BMM2 op constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t v_dim [4] =  {b, h, s_kv, d};
    int64_t v_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);

    int64_t o_dim [4] =  {b, h, s_q, d};
    int64_t o_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);

    auto vTensor = tensor_create(tensorType, V_ID, v_dim, v_stride, false, false);
    // second GEMM output
    auto oTensor = tensor_create(tensorType, O_ID, o_dim, o_stride, false, false);

    // Define the matmul 2 desc
    auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();

    // Create a matmul 2 Node
    auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(afterScaleDropoutTensor)
                            .setbMatDesc(vTensor)
                            .setcMatDesc(oTensor)
                            .setmatmulDesc(matmul_2_Desc)
                            .build();


    ops.push_back(std::move(matmul_op2));
}

#endif

} // namespace

void run_mha_plan(cudnnHandle_t handle,
		  cudnn_frontend::ExecutionPlan& plan,
		  void* devPtrQ,
		  void* devPtrK,
		  void* devPtrV,
		  void* devPtrO,
		  void* devPtrDropoutSeed,
          void* devPtrDropoutOffset,
		  void* devPtrSoftmaxStats,
		  float scale_dropout,
		  float scaling_factor,
		  bool is_training) {

    auto workspace_size = plan.getWorkspaceSize();

    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace_ptr = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size).get();
    }

    std::set<std::pair<uint64_t, void*>> data_ptrs;
    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, void*>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, void*>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, void*>(V_ID, devPtrV));
    // Causual mask
    float negInfinity = -1.0E+20f; // change this if you have access to float_min

    data_ptrs.insert(std::pair<uint64_t, void*>(MASK_VAL_ID, &negInfinity));
    data_ptrs.insert(std::pair<uint64_t, void*>(S_CONST_ID, &scaling_factor));
    data_ptrs.insert(std::pair<uint64_t, void*>(O_ID, devPtrO));
    data_ptrs.insert(std::pair<uint64_t, void*>(D_SEED_ID, devPtrDropoutSeed));
    data_ptrs.insert(std::pair<uint64_t, void*>(D_OFFSET_ID, devPtrDropoutOffset));
    data_ptrs.insert(std::pair<uint64_t, void*>(D_CONST_ID, &scale_dropout));

    // If training mode, we write out softmax stats
    if (is_training) {
        data_ptrs.insert(std::pair<uint64_t, void*>(S_STATS_ID, devPtrSoftmaxStats));
    }

    auto variantPack  = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace_ptr)
                           .setDataPointers(data_ptrs)
                           .build();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}


void
run_bf16_LLM_fprop(int64_t b,
              int64_t h,
              int64_t s_q,
              int64_t s_kv,
              int64_t d,
              MHA_Layout layout,
              float scaling_factor,
              bool is_training,
              double dropout_probability,
              void* devPtrQ,
              void* devPtrK,
              void* devPtrV,
              void* devPtrSoftmaxStats,
              void* devPtrO,
              void* devPtrDropoutSeed,
              void* devPtrDropoutOffset,
              cudnnDataType_t tensorType) {

    try {
        cudnnHandle_t handle_ = getCudnnHandle();
	MHAParams params;
	setMHAParams(params, b, h, s_q, s_kv, d, layout, is_training, tensorType);
	// needs to be bf16 (Please change)
	__half scale_dropout = cpu_float2half_rn(static_cast<float>(1/(1 - dropout_probability)));

	auto search = benchmark_cache.find(params);
	if (!search) {
		std::vector<cudnn_frontend::Operation const*> all_ops;
		std::vector<cudnn_frontend::Operation> ops;

		// Q * K^T
		auto sTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops);

		// Q * K^T * bmmScale
		auto sScaleTensor = createScale(b, h, s_q, s_kv, d, layout, CUDNN_DATA_FLOAT, sTensor, ops);

		auto sAfterMaskTensor = createCausalMask(b, h, s_q, s_kv, d, layout, tensorType, ops, sScaleTensor);

		// cudnn_frontend::throw_if(dropout_probability != 0.0f && !is_training, "Dropout probability should be 0.0f for inference mode", CUDNN_STATUS_BAD_PARAM);
		cudnn_frontend::throw_if(dropout_probability == 1.0f, "Dropout probability cannot be 1.0", CUDNN_STATUS_BAD_PARAM);

		auto softmax_output = createSoftmaxForward(b, h, s_q, s_kv, is_training, ops, sAfterMaskTensor);

		// Dropout(softmax)
		auto dropout_output = createDropoutForward(b, h, s_q, s_kv, d, dropout_probability, tensorType, ops, softmax_output);
		createSVBMM(b, h, s_q, s_kv, d, layout, tensorType, ops, dropout_output);

		for (unsigned int i = 0; i < ops.size(); i++) {
		    all_ops.push_back(&ops[i]);
		}

		// Create an Operation Graph
		auto opGraph = cudnn_frontend::OperationGraphBuilder()
				   .setHandle(handle_)
				   .setOperationGraph(all_ops.size(), all_ops.data())
				   .build();


		cudnn_frontend::EngineConfigList filtered_configs;
		auto statuses = cudnn_frontend::get_heuristics_list<1>({"heuristics_instant"}, opGraph, allowAllConfig, filtered_configs, true);

		if (filtered_configs.size() == 0) {
		    cudnn_frontend::set_error_and_throw_exception(
			    nullptr,
			    CUDNN_STATUS_NOT_SUPPORTED,
			    "run_mha_fprop: No config returned by the heuristics");
		}
            auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
	    run_mha_plan(handle_,
		         plan,
		         devPtrQ,
		         devPtrK,
		         devPtrV,
		         devPtrO,
		         devPtrDropoutSeed,
                         devPtrDropoutOffset,
			 devPtrSoftmaxStats,
		         scale_dropout,
		         scaling_factor,
			 is_training);
	    benchmark_cache.update(params, plan);
	} else {
	    run_mha_plan(handle_,
		         *search,
		         devPtrQ,
		         devPtrK,
		         devPtrV,
		         devPtrO,
		         devPtrDropoutSeed,
                         devPtrDropoutOffset,
			 devPtrSoftmaxStats,
		         scale_dropout,
		         scaling_factor,
			 is_training);
       }
    } catch (cudnn_frontend::cudnnException& e) {
        TORCH_WARN("cudnnException ", e.what());
    }
}

// END COPY-PASTE FRO CUDNN-FRONTEND SAMPLE

void
run_cudnn_LLM_fprop(int64_t b,
                    int64_t h,
                    int64_t s_q,
                    int64_t s_kv,
                    int64_t d,
                    float scaling_factor,
                    bool return_softmaxstats,
                    double dropout_probability,
                    const Tensor& q,
                    const Tensor& k,
                    const Tensor& v,
                    Tensor& softmaxstats,
                    Tensor& o,
                    Tensor& dropoutseed,
                    Tensor& dropoutoffset) {
    cudnnDataType_t tensorType = CUDNN_DATA_HALF;
    if (q.scalar_type() == kBFloat16) {
      tensorType = CUDNN_DATA_BFLOAT16;
    }
    o = at::zeros({b, s_q, h, d}, q.options());
    if (return_softmaxstats) {
      softmaxstats = at::zeros({b, h, s_q}, q.options());
    }
    run_bf16_LLM_fprop( b,
                   h,
                   s_q,
                   s_kv,
                   d,
                  MHA_Layout::QKV_INTERLEAVED,
                  scaling_factor,
                  return_softmaxstats,
                  dropout_probability,
                  q.data_ptr(),
                  k.data_ptr(),
                  v.data_ptr(),
                  return_softmaxstats ? softmaxstats.data_ptr() : nullptr,
                  o.data_ptr(),
                  dropoutseed.data_ptr(),
                  dropoutoffset.data_ptr(),
                  tensorType);
}

Tensor cudnn_mha(const Tensor& self, const Tensor& qkv_weight, const Tensor& out_weight, const Tensor& qkv_bias, const Tensor& out_bias, const long inputSize, const long seqLength, const long batchSize, const long outputSize, const long head_size, const long num_heads) {
  AT_ERROR("cudnn_mha: ATen not compiled with cuDNN support");
}

}} // namespace at::native

#endif
