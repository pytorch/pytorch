#pragma once

#include <c10/util/irange.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/perfkernels/embedding_lookup.h"
#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
#endif
#include <algorithm>
#include <functional>

namespace caffe2 {

// A templated class that implements SparseLengths[Sum,WeightedSum,Mean].
template <
    typename T, // output type
    class InputTypes, // supported input types, such as TensorTypes<float>
    bool USE_WEIGHT = false, // Whether it is SparseLengthsWeightedSum
    bool USE_MEAN = false, // Whether this is SparseLengthsMean
    bool USE_POSITIONAL_WEIGHT = false
    // USE_WEIGHT = true and USE_POSITIONAL_WEIGHT = true
    // -> SparseLengthsPositionalWeightedSum
    >
class CPUSparseLengthsReductionOp : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  template <class... Args>
  explicit CPUSparseLengthsReductionOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {
    static_assert(
        !(USE_WEIGHT & USE_MEAN), "Cannot both specify weight and mean.");
  }

  ~CPUSparseLengthsReductionOp() {}

  // Currently, we support float and at::Half inputs for input data type, and
  // int32_t and int64_t for the index type.

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(DATA));
  }

  template <typename InputType>
  bool DoRunWithType() {
    return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
        this, Input(INDICES));
  }

  template <typename InputType, typename IndexType>
  bool DoRunWithType2() {
    auto& dataInput = Input(DATA);
    auto& indicesInput = Input(INDICES);
    auto& lengthsInput = Input(LENGTHS);

    const int64_t M = lengthsInput.size(0);
    const int64_t indices_size = indicesInput.numel();

    auto shape = dataInput.sizes().vec();
    shape[0] = M;
    auto* output = Output(0, shape, at::dtype<T>());
    T* out_data = output->template mutable_data<T>();

    if (indices_size == 0) {
      if (M > 0) {
        memset(out_data, 0, output->numel() * sizeof(T));
      }
      return true;
    }

    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t N = dataInput.size(0);
    const int D = dataInput.size_from_dim(1);

    const InputType* in_data = dataInput.template data<InputType>();
    const IndexType* indices = indicesInput.template data<IndexType>();
    const int* lengths = lengthsInput.template data<int>();
    const T* in_weight = nullptr;

    if (USE_WEIGHT) {
      // static if
      auto& weightInput = Input(WEIGHT);
      CAFFE_ENFORCE_EQ(1, weightInput.dim(), "WEIGHT must be a vector");
      if (!USE_POSITIONAL_WEIGHT) {
        CAFFE_ENFORCE_EQ(
            weightInput.numel(),
            indices_size,
            "Weight should have the same length as indices.");
      }
      in_weight = weightInput.template data<T>();
    }

#ifdef USE_FBGEMM
    // If this is the first call or block size has changed (should never
    // happen actually), generate a kernel.
    if (D != last_block_size) {
      last_block_size = D;
      if (std::is_same<InputType, float>::value) {
        if (std::is_same<IndexType, std::int32_t>::value) {
          kernel_fp32_i32_ =
              fbgemm::GenerateEmbeddingSpMDM<float, std::int32_t>(
                  D,
                  USE_WEIGHT,
                  USE_MEAN,
                  /*prefetch distance*/ 16,
                  USE_POSITIONAL_WEIGHT,
                  /*use_offsets*/ false);
        } else {
          CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
          kernel_fp32_i64_ =
              fbgemm::GenerateEmbeddingSpMDM<float, std::int64_t>(
                  D,
                  USE_WEIGHT,
                  USE_MEAN,
                  /*prefetch distance*/ 16,
                  USE_POSITIONAL_WEIGHT,
                  /*use_offsets*/ false);
        }
      } else {
        CAFFE_ENFORCE((std::is_same<InputType, at::Half>::value));
        if (std::is_same<IndexType, std::int32_t>::value) {
          kernel_fp16_i32_ =
              fbgemm::GenerateEmbeddingSpMDM<fbgemm::float16, std::int32_t>(
                  D,
                  USE_WEIGHT,
                  USE_MEAN,
                  /*prefetch distance*/ 16,
                  USE_POSITIONAL_WEIGHT,
                  /*use_offsets*/ false);
        } else {
          CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
          kernel_fp16_i64_ =
              fbgemm::GenerateEmbeddingSpMDM<fbgemm::float16, std::int64_t>(
                  D,
                  USE_WEIGHT,
                  USE_MEAN,
                  /*prefetch distance*/ 16,
                  USE_POSITIONAL_WEIGHT,
                  /*use_offsets*/ false);
        }
      }
    }

    bool success;
    if (std::is_same<InputType, float>::value) {
      if (std::is_same<IndexType, std::int32_t>::value) {
        success = kernel_fp32_i32_(
            M,
            indices_size,
            N,
            reinterpret_cast<const float*>(in_data),
            indicesInput.template data<std::int32_t>(),
            lengths,
            in_weight,
            out_data);
      } else {
        success = kernel_fp32_i64_(
            M,
            indices_size,
            N,
            reinterpret_cast<const float*>(in_data),
            indicesInput.template data<std::int64_t>(),
            lengths,
            in_weight,
            out_data);
      }
    } else {
      if (std::is_same<IndexType, std::int32_t>::value) {
        success = kernel_fp16_i32_(
            M,
            indices_size,
            N,
            reinterpret_cast<const fbgemm::float16*>(in_data),
            indicesInput.template data<std::int32_t>(),
            lengths,
            in_weight,
            out_data);
      } else {
        success = kernel_fp16_i64_(
            M,
            indices_size,
            N,
            reinterpret_cast<const fbgemm::float16*>(in_data),
            indicesInput.template data<std::int64_t>(),
            lengths,
            in_weight,
            out_data);
      }
    }

    if (success) {
      return true;
    }

    int64_t current = 0;
    for (const auto m : c10::irange(M)) {
      for (int i = 0; i < lengths[m]; ++i) {
        CAFFE_ENFORCE_LT(
            current,
            indices_size,
            "Your input seems to be incorrect: the sum of lengths values "
            "should be the size of the indices tensor, but it appears not.");
        IndexType idx = indices[current];
        CAFFE_ENFORCE(
            0 <= idx && idx < N,
            "Index ",
            current,
            " is out of bounds: ",
            idx,
            ", range 0 to ",
            N,
            ", actual batch length is ",
            M);
        ++current;
      }
    }
    CAFFE_ENFORCE_EQ(
        current,
        indices_size,
        "Your input seems to be incorrect: the sum of lengths values should be "
        "the size of the indices tensor, but it appears not.");

    return false;
#endif

    // delegate work to perfkernel that branches based on architecture
    EmbeddingLookup<IndexType, InputType, T, USE_POSITIONAL_WEIGHT>(
        D,
        M,
        indices_size,
        N,
        in_data,
        indices,
        lengths,
        in_weight,
        nullptr, // scale_bias field is only used in SparseLengths8BitsRowwiseOp
        USE_MEAN,
        out_data);
    return true;
  }

  enum {
    DATA = 0, // Data input.
    WEIGHT = 1, // Weight input used in SparseLengthsWeightedSum
    INDICES = 1 + USE_WEIGHT, // 1 in SparseLengths[Sum,Mean] and
                              // 2 in SparseLengthsWeightedSum
    LENGTHS = 2 + USE_WEIGHT, // 2 in SparseLengths[Sum, Mean],
                              // 3 in SparseLengthsWeightedSum
  };

#ifdef USE_FBGEMM
 private:
  std::int64_t last_block_size{-1};
  fbgemm::EmbeddingSpMDMKernelSignature<float, std::int32_t>::Type
      kernel_fp32_i32_;
  fbgemm::EmbeddingSpMDMKernelSignature<float, std::int64_t>::Type
      kernel_fp32_i64_;
  fbgemm::EmbeddingSpMDMKernelSignature<fbgemm::float16, std::int32_t>::Type
      kernel_fp16_i32_;
  fbgemm::EmbeddingSpMDMKernelSignature<fbgemm::float16, std::int64_t>::Type
      kernel_fp16_i64_;
#endif
};

template <typename T, class Context, class Engine = DefaultEngine>
class TTSparseLengthsSumOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit TTSparseLengthsSumOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        factor_i(this->template GetRepeatedArgument<int>(
            "factor_i",
            vector<int>{1, 1, 1})),
        factor_j(this->template GetRepeatedArgument<int>(
            "factor_j",
            vector<int>{1, 1, 1})),
        ranks(this->template GetRepeatedArgument<int>(
            "ranks",
            vector<int>{1, 1, 1, 1})),
        emb_size(this->template GetSingleArgument<int>("emb_size", 64)) {
    // cumprod of i, used for index slice
    l_cumprod.push_back(1);
    for (const auto i : c10::irange(1, factor_i.size())) {
      l_cumprod.push_back(l_cumprod[i - 1] * factor_i[i - 1]);
    }
  }

  ~TTSparseLengthsSumOp() {}

  void Ind2Sub(int64_t* out_factor_index, const int64_t* indices, int len) {
    // TODO: vectorization
    auto N = factor_i.size();
    for (const auto j : c10::irange(len)) {
      auto idx = indices[j];
      for (int i = N; i > 0; i--) {
        out_factor_index[j * N + i - 1] = idx / l_cumprod[i - 1];
        idx = idx % l_cumprod[i - 1];
      }
    }
  }

  bool GetSlice(
      std::vector<std::vector<T>>& tgt_slice,
      const T* core,
      const vector<int64_t>& ind_slice,
      int bs,
      int idx) {
    // implement the functinality index_select(core, 1, ind_slice)
    auto num_of_elements = ranks[idx] * factor_j[idx] * ranks[idx + 1];
    for (const auto i : c10::irange(bs)) {
      memcpy(
          tgt_slice[i].data(),
          core + ind_slice[i] * num_of_elements,
          num_of_elements * sizeof(T));
    }
    return true;
  }

  // ind: it stores the index to each tensor core
  // bs: the number of indices
  // GatherAllRows uses two steps to calculate the lengthsum functionality: 1) it uses tensor train
  // to calculate the embedding for each index. 2) it sums the embedding for each bag.
  // In Step 1), it batches all the indices together. Specifically, for every index, it uses the pre-computed
  // ind of each tensor core to extract the corresponding slice of the core. Then it does gemm operation
  // sequentially on the slices to produce the embedding result for each index.
  // In Step 2), it takes the embedding computed in step 1) and apply the sum operation for each bag.
  bool GatherAllRows(
      int64_t* ind,
      int bs,
      int x_len,
      vector<const T*> cores,
      int segments,
      const int* lengths,
      T* out_data) {
    // compute the largest memory consumption of intermediate result
    // TODO: dynamic allocation size: cur_rows*factor_j[i]*ranks[i+1]
    // and also explore the contiguous memory storage for res and int_res
    int max_rank = *max_element(ranks.begin(), ranks.end());
    std::vector<std::vector<T>> res(bs, std::vector<T>(emb_size * max_rank, 0));
    std::vector<std::vector<T>> int_res(
        bs, std::vector<T>(emb_size * max_rank, 0));

    // Store the matrix A
    vector<T*> Y_ptr(bs);
    // Store the intermediate result in each layer
    vector<T*> Z_ptr(bs);

    for (const auto b : c10::irange(bs)) {
      Y_ptr[b] = res[b].data();
      Z_ptr[b] = int_res[b].data();
    }

    vector<int64_t> ind_slice(bs);
    int rows = 0;
    for (const auto i : c10::irange(x_len)) {
      // slice cur
      for (const auto j : c10::irange(bs)) {
        ind_slice[j] = ind[x_len * j + i];
      }
      if (i == 0) {
        GetSlice(res, cores[i], ind_slice, bs, i);
        rows = factor_j[0];
      } else {
        std::vector<std::vector<T>> slice(
            bs, std::vector<T>(ranks[i] * factor_j[i] * ranks[i + 1], 0));
        vector<const T*> X_ptr(bs);
        for (const auto b : c10::irange(bs)) {
          X_ptr[b] = slice[b].data();
        }
        GetSlice(slice, cores[i], ind_slice, bs, i);

        math::GemmBatched<T, CPUContext>(
            CblasNoTrans,
            CblasNoTrans,
            bs,
            rows,
            factor_j[i] * ranks[i + 1],
            ranks[i],
            1.0f,
            const_cast<const T**>(Y_ptr.data()),
            X_ptr.data(),
            0.0f,
            Z_ptr.data(),
            &context_);
        for (const auto b : c10::irange(bs)) {
          std::memcpy(Y_ptr[b], Z_ptr[b], (emb_size * max_rank) * sizeof(T));
        }
        rows *= factor_j[i];
      }
      // save the intermediate output for backward path
      // shape for the core
      auto shape = vector<int64_t>({bs, rows, ranks[i + 1]});
      if (i < 2) {
        auto* core_data = Output(i + 1, shape, at::dtype<T>());
        T* out_core = core_data->template mutable_data<T>();
        for (const auto b : c10::irange(bs)) {
          std::memcpy(
              out_core + b * rows * ranks[i + 1],
              Y_ptr[b],
              rows * ranks[i + 1] * sizeof(T));
        }
      }
    }

    // reduction and store back to output
    vector<int64_t> cum_lengths(segments);
    for (const auto seg : c10::irange(segments)) {
      cum_lengths[seg] =
          seg == 0 ? lengths[0] : lengths[seg] + cum_lengths[seg - 1];
    }

    int length_idx = 0;
    vector<T> tmp_sum(emb_size, 0.0f);
    for (int i = 0; i <= bs; i++) {
      while ((length_idx < segments) && (i == cum_lengths[length_idx])) {
        // store the tmp_sum into output
        memcpy(
            &out_data[length_idx * emb_size],
            tmp_sum.data(),
            emb_size * sizeof(T));
        length_idx++;
        fill(tmp_sum.begin(), tmp_sum.end(), 0.0f);
      }
      if (i == bs) {
        break;
      }
      transform(
          res[i].begin(),
          res[i].begin() + emb_size,
          tmp_sum.begin(),
          tmp_sum.begin(),
          std::plus<T>());
    }
    return true;
  }

  bool RunOnDevice() override {
    const auto& dataInput0 = Input(0);
    const auto& dataInput1 = Input(1);
    const auto& dataInput2 = Input(2);
    const auto& indicesInput = Input(3);
    const auto& lengthsInput = Input(4);

    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");

    int N = factor_i.size();
    const int64_t M = lengthsInput.size(0);

    auto shape = vector<int64_t>({M, emb_size});
    auto* output = Output(0, shape, at::dtype<T>());
    T* out_data = output->template mutable_data<T>();

    const T* core0 = dataInput0.template data<T>();
    const T* core1 = dataInput1.template data<T>();
    const T* core2 = dataInput2.template data<T>();

    const int* lengths = lengthsInput.template data<int>();

    vector<const T*> cores = {core0, core1, core2};

    const int64_t* indices = indicesInput.template data<int64_t>();

    // Store the factor index for backward path
    auto index_shape = vector<int64_t>({indicesInput.size(), N});
    auto* index_data = Output(3, index_shape, at::dtype<int64_t>());
    int64_t* out_factor_index = index_data->template mutable_data<int64_t>();

    // Store the factorized index for each core
    Ind2Sub(out_factor_index, indices, indicesInput.size());

    return GatherAllRows(
        out_factor_index, indicesInput.size(), N, cores, M, lengths, out_data);
  }

 protected:
  vector<int> factor_i;
  vector<int> factor_j;
  vector<int> ranks;
  vector<int> l_cumprod;
  int emb_size;
};

template <typename T, class Context>
class TTSparseLengthsSumGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit TTSparseLengthsSumGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  bool RunOnDevice() override;

  ~TTSparseLengthsSumGradientOp() {}
};

// implement the graident op for TTLengthSumGradient op
template <typename T, class Context>
bool TTSparseLengthsSumGradientOp<T, Context>::RunOnDevice() {
  const auto& core0 = Input(0);
  const auto& core1 = Input(1);
  const auto& core2 = Input(2);
  const auto& lengths = Input(3);
  const auto& core0_out = Input(4);
  const auto& core1_out = Input(5);
  const auto& index_out = Input(6);
  const auto& dY = Input(7);

  const int* lengths_data = lengths.template data<int>();
  const T* dY_data = dY.template data<T>();

  // restore the arguments from shape
  const int64_t bs = index_out.size(0);
  const int64_t emb_size = dY.size(1);
  const int64_t num_segments = lengths.size(0);

  auto core0_shape = core0.sizes().vec();
  auto core1_shape = core1.sizes().vec();
  auto core2_shape = core2.sizes().vec();
  auto core0_out_shape = core0_out.sizes().vec();
  auto core1_out_shape = core1_out.sizes().vec();

  auto* dCore0 = Output(0, core0_shape, at::dtype<T>());
  auto* dCore1 = Output(1, core1_shape, at::dtype<T>());
  auto* dCore2 = Output(2, core2_shape, at::dtype<T>());

  T* dCore0_data = dCore0->template mutable_data<T>();
  T* dCore1_data = dCore1->template mutable_data<T>();
  T* dCore2_data = dCore2->template mutable_data<T>();

  memset(
      dCore0_data,
      0.0f,
      sizeof(T) *
          accumulate(
              core0_shape.begin(), core0_shape.end(), 1, std::multiplies<T>()));
  memset(
      dCore1_data,
      0.0f,
      sizeof(T) *
          accumulate(
              core1_shape.begin(), core1_shape.end(), 1, std::multiplies<T>()));
  memset(
      dCore2_data,
      0.0f,
      sizeof(T) *
          accumulate(
              core2_shape.begin(), core2_shape.end(), 1, std::multiplies<T>()));

  int64_t* index_out_data = index_out.template mutable_data<int64_t>();

  vector<vector<int64_t>> index_slice(bs, vector<int64_t>(3, 0));
  for (const auto b : c10::irange(bs)) {
    memcpy(index_slice[b].data(), index_out_data + b * 3, 3 * sizeof(int64_t));
  }

  vector<const T*> A_ptr(bs);
  vector<T*> B_ptr(bs);
  vector<T*> C_ptr(bs);
  // size of each batch
  int64_t num_of_elements = 0;

  // construct the ranks
  // expand the gradient into all indices
  vector<vector<T>> core2_out_grad(bs, vector<T>(emb_size, 0));
  int64_t data_index = 0;
  for (const auto range_index : c10::irange(num_segments)) {
    for (int64_t start = data_index;
         data_index < start + lengths_data[range_index];
         ++data_index) {
      memcpy(
          core2_out_grad[data_index].data(),
          dY_data + range_index * emb_size,
          emb_size * sizeof(T));
    }
  }

  // =======================================================
  // Calculate dCore2_data:
  // 1) Transpose core1_out and multiply iwth core2_out_grad
  // 2)  add to dCore2_data
  vector<vector<T>> dCore2_data_slice_grad(
      bs, vector<T>(core2_shape[1] * core2_shape[2] * core2_shape[3], 0));
  const T* core1_out_data = core1_out.template data<T>();
  // const T* core1_out_p[bs];
  for (const auto b : c10::irange(bs)) {
    A_ptr[b] = core1_out_data + b * core1_out.size(1) * core1_out.size(2);
    B_ptr[b] = core2_out_grad[b].data();
    C_ptr[b] = dCore2_data_slice_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasTrans,
      CblasNoTrans,
      bs,
      core2.size(1), // M
      core2.size(2) * core2.size(3), // N
      core1_out.size(1), // K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_);

  // update the corresponding slice
  num_of_elements = core2_shape[1] * core2_shape[2] * core2_shape[3];

  T* core2_data = core2.template mutable_data<T>();
  vector<vector<T>> core2_slice(
      bs, vector<T>(core2_shape[1] * core2_shape[2] * core2_shape[3], 0));

  for (const auto b : c10::irange(bs)) {
    for (const auto i : c10::irange(num_of_elements)) {
      dCore2_data[index_slice[b][2] * num_of_elements + i] += C_ptr[b][i];
    }
    memcpy(
        core2_slice[b].data(),
        core2_data + index_slice[b][2] * num_of_elements,
        sizeof(T) * num_of_elements);
  }

  // Calculate core1_out_grad
  vector<vector<T>> core1_out_grad(
      bs, vector<T>(core1_out_shape[1] * core1_out_shape[2], 0));

  for (const auto b : c10::irange(bs)) {
    A_ptr[b] = core2_out_grad[b].data();
    B_ptr[b] = core2_slice[b].data();
    C_ptr[b] = core1_out_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasNoTrans,
      CblasTrans,
      bs,
      core1_out.size(1), // M
      core2_shape[1], // N
      core2_shape[2] * core2_shape[3], // K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_);

  // =======================================================
  // Calcuate dCore1_data:
  // 1) Transpose core1_out_grad and multiply with core0_out
  // 2) Transpose the result and then add to dCore1_data
  vector<vector<T>> dCore1_data_slice_grad(
      bs, vector<T>(core1_shape[1] * core1_shape[2] * core1_shape[3], 0));
  const T* core0_out_data = core0_out.template data<T>();
  for (const auto b : c10::irange(bs)) {
    A_ptr[b] = core0_out_data + b * core0_out.size(1) * core0_out.size(2);
    B_ptr[b] = core1_out_grad[b].data();
    C_ptr[b] = dCore1_data_slice_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasTrans,
      CblasNoTrans,
      bs,
      core1.size(1), // M
      core1.size(2) * core1.size(3), // N
      core0_out.size(1), // K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_);

  // update the corresponding slice
  num_of_elements = core1_shape[1] * core1_shape[2] * core1_shape[3];
  T* core1_data = core1.template mutable_data<T>();
  vector<vector<T>> core1_slice(
      bs, vector<T>(core1_shape[1] * core1_shape[2] * core1_shape[3], 0));

  for (const auto b : c10::irange(bs)) {
    for (const auto i : c10::irange(num_of_elements)) {
      dCore1_data[index_slice[b][1] * num_of_elements + i] += C_ptr[b][i];
    }
    memcpy(
        core1_slice[b].data(),
        core1_data + index_slice[b][1] * num_of_elements,
        sizeof(T) * num_of_elements);
  }

  // Calcuate core0_out_grad
  vector<vector<T>> core0_out_grad(
      bs, vector<T>(core0_out_shape[1] * core0_out_shape[2], 0));

  for (const auto b : c10::irange(bs)) {
    A_ptr[b] = core1_out_grad[b].data();
    B_ptr[b] = core1_slice[b].data();
    C_ptr[b] = core0_out_grad[b].data();
  }

  math::GemmBatched<T, CPUContext>(
      CblasNoTrans,
      CblasTrans,
      bs,
      core0_out.size(1), // M
      core1_shape[1], // N
      core1_shape[2] * core1_shape[3], // K
      1.0f,
      const_cast<const T**>(A_ptr.data()),
      const_cast<const T**>(B_ptr.data()),
      0.0f,
      C_ptr.data(),
      &context_);

  num_of_elements = core0_shape[1] * core0_shape[2] * core0_shape[3];

  for (const auto b : c10::irange(bs)) {
    for (const auto i : c10::irange(num_of_elements)) {
      dCore0_data[index_slice[b][0] * num_of_elements + i] += C_ptr[b][i];
    }
  }
  return true;
}

} // namespace caffe2
