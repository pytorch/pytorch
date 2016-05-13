#pragma once

#include "caffe2/utils/math.h"
#include "caffe2/core/context.h"

namespace caffe2 {

////////////////////////////////////////////////////////////////////////////////
// Range reducers: can leverage that input segment is continuous and provide
// special implementation
////////////////////////////////////////////////////////////////////////////////

// Put forward and backward in the same template?
template <typename T, class Context>
class SumRangeReducer;
template <typename T, class Context>
class SumRangeReducerGradient;

template <typename T>
class SumRangeReducer<T, CPUContext> {
 public:
  void operator()(
      const TIndex block_size,
      const TIndex blocks,
      const T* in,
      T* out,
      CPUContext* context) {
    // do we need to go through wrapper in math.h?
    EigenVectorMap<T> out_vec(out, block_size);
    out_vec = ConstEigenMatrixMap<T>(in, block_size, blocks).rowwise().sum();
  }
};

template <typename T, class Context>
class SumRangeReducerGradient {
 public:
  void operator()(
      const TIndex block_size,
      const TIndex blocks,
      const T* segment_grad,
      T* data_grad,
      Context* context) {
    // do we have some op that does it smartly with minimum number of memcpy?
    for (TIndex i = 0; i < blocks; ++i) {
      context->template Copy<T, Context, Context>(
          block_size, segment_grad, data_grad + block_size * i);
    }
  }
};

struct SumRangeReducerDef {
  template <typename T, class Context>
  using Reducer = SumRangeReducer<T, Context>;
  template <typename T, class Context>
  using ReducerGradient = SumRangeReducerGradient<T, Context>;
  static constexpr const char* name = "Sum";
};

////////////////////////////////////////////////////////////////////////////////
// Incremental reducers: consume elements one by one
////////////////////////////////////////////////////////////////////////////////

// Put forward and backward in the same template?
template <typename T, class Context>
class SumReducer;
template <typename T, class Context>
class SumReducerGradient;

template <typename T>
class SumReducer<T, CPUContext> {
 public:
  static constexpr int kInputCount = 1;

  struct Meta {
    TIndex block_size;
    vector<TIndex> block_shape;

    void
    observeInput(int input, const Tensor<CPUContext>& value, int skip_dims) {
      CAFFE_DCHECK_EQ(0, input);
      auto& dims = value.dims();
      block_shape.assign(dims.begin() + skip_dims, dims.end());
      block_size = value.size_from_dim(skip_dims);
    }

    void appendOutputShape(vector<TIndex>* output_shape) {
      output_shape->insert(
          output_shape->end(), block_shape.begin(), block_shape.end());
    }
  };

  SumReducer(const Meta& meta, T* out, CPUContext* context) : out_(out) {
    // add a wrapper in Context for it
    memset(out, 0, sizeof(T) * meta.block_size);
  }
  void
  process(const Meta& meta, const T* in, TIndex offset, CPUContext* context) {
    math::Axpy<T>(meta.block_size, 1, in, out_, context);
  }

 private:
  T* out_;
};

template <typename T, class Context>
class SumReducerGradient {
 public:
  // which of the original inputs are required for gradient computation
  static constexpr std::array<int, 0> originalInputs() {
    return {};
  }

  struct Meta {
    TIndex block_size;
    vector<TIndex> block_shape;

    Meta(const Tensor<CPUContext>& out_grad, int skip_dims) {
      auto& dims = out_grad.dims();
      block_shape.assign(dims.begin() + skip_dims, dims.end());
      block_size = out_grad.size_from_dim(skip_dims);
    }

    void observeOriginalInput(
        int original_input,
        const Tensor<CPUContext>& value,
        int skip_dims) {}

    void appendGradShape(vector<TIndex>* output_shape) {
      output_shape->insert(
          output_shape->end(), block_shape.begin(), block_shape.end());
    }
  };

  SumReducerGradient(const Meta& meta, const T* s_grad, CPUContext* context)
      : s_grad_(s_grad) {}

  void
  fillGrad(const Meta& meta, T* data_grad, TIndex offset, Context* context) {
    context->template Copy<T, Context, Context>(
        meta.block_size, s_grad_, data_grad);
  }

 private:
  const T* s_grad_;
};

struct SumReducerDef {
  template <typename T, class Context>
  using Reducer = SumReducer<T, Context>;
  template <typename T, class Context>
  using ReducerGradient = SumReducerGradient<T, Context>;
  static constexpr const char* name = "Sum";
};

// Put forward and backward in the same template?
template <typename T, class Context>
class WeightedSumReducer;
template <typename T, class Context>
class WeightedSumReducerGradient;

template <typename T>
class WeightedSumReducer<T, CPUContext> {
 public:
  static constexpr int kInputCount = 2;

  struct Meta {
    TIndex block_size;
    vector<TIndex> block_shape;
    const T* scalars;

    void
    observeInput(int input, const Tensor<CPUContext>& value, int skip_dims) {
      if (input == 1) {
        CAFFE_CHECK_EQ(skip_dims, value.ndim())
            << "SCALARS mustn't have extra dimensions";
        scalars = value.data<T>();
        return;
      }
      CAFFE_DCHECK_EQ(0, input);
      auto& dims = value.dims();
      block_shape.assign(dims.begin() + skip_dims, dims.end());
      block_size = value.size_from_dim(skip_dims);
    }

    void appendOutputShape(vector<TIndex>* output_shape) {
      output_shape->insert(
          output_shape->end(), block_shape.begin(), block_shape.end());
    }
  };

  WeightedSumReducer(const Meta& meta, T* out, CPUContext* context)
      : out_(out) {
    // do we have a wrapper for it?
    memset(out, 0, sizeof(T) * meta.block_size);
  }
  void
  process(const Meta& meta, const T* in, TIndex offset, CPUContext* context) {
    math::Axpy<T>(meta.block_size, meta.scalars[offset], in, out_, context);
  }

 private:
  T* out_;
};

template <typename T, class Context>
class WeightedSumReducerGradient {
 public:
  // which of the original inputs are required for gradient computation
  static constexpr std::array<int, 1> originalInputs() {
    return {1};
  }

  struct Meta {
    TIndex block_size;
    vector<TIndex> block_shape;
    const T* scalars;

    Meta(const Tensor<CPUContext>& out_grad, int skip_dims) {
      auto& dims = out_grad.dims();
      block_shape.assign(dims.begin() + skip_dims, dims.end());
      block_size = out_grad.size_from_dim(skip_dims);
    }

    void observeOriginalInput(
        int original_input,
        const Tensor<CPUContext>& value,
        int skip_dims) {
      CAFFE_CHECK_EQ(1, original_input);
      scalars = value.data<T>();
    }

    void appendGradShape(vector<TIndex>* output_shape) {
      output_shape->insert(
          output_shape->end(), block_shape.begin(), block_shape.end());
    }
  };

  WeightedSumReducerGradient(
      const Meta& meta,
      const T* s_grad,
      CPUContext* context)
      : s_grad_(s_grad) {}

  void
  fillGrad(const Meta& meta, T* data_grad, TIndex offset, Context* context) {
    math::Scale(
        meta.block_size, meta.scalars[offset], s_grad_, data_grad, context);
  }

 private:
  const T* s_grad_;
};

struct WeightedSumReducerDef {
  template <typename T, class Context>
  using Reducer = WeightedSumReducer<T, Context>;
  template <typename T, class Context>
  using ReducerGradient = WeightedSumReducerGradient<T, Context>;
  static constexpr const char* name = "WeightedSum";
};

}
