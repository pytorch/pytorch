#ifndef CAFFE2_OPERATORS_TILE_OP_H_
#define CAFFE2_OPERATORS_TILE_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Copy a Blob n times along a specified axis.
template <class Context>
class TileOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TileOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        tiles_(OperatorBase::GetSingleArgument<int32_t>("tiles", 1)),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 0)) {}
  ~TileOp() {}

  bool RunOnDevice() override {
    const auto& input = Input(0);
    std::array<int32_t, 2> temp_params = {{tiles_, axis_}};
    if (InputSize() > 1) {
      // We potentially have tiles and/or axis specified as inputs
      // as well. We will check for them in that order. In other words:
      // InputSize() == 2: tiles is specified
      // InputSize() == 3: tiles is specified and axis.
      // Anything specified as input will override the arguments
      CAFFE_ENFORCE(
          Input(1).ndim() == 1 && Input(1).size() == 1,
          "Input `tiles` should be a vector of size 1.");

      const auto& input1 = Input(1);
      context_.template CopyItems<Context, CPUContext>(
          input1.meta(),
          1,
          static_cast<const char*>(input1.raw_data()),
          &(temp_params[0]));

      if (InputSize() > 2) {
        CAFFE_ENFORCE(
            Input(2).ndim() == 1 && Input(2).size() == 1,
            "Input `axis` should be a vector of size 1.");

        const auto& input2 = Input(2);
        context_.template CopyItems<Context, CPUContext>(
            input2.meta(),
            1,
            static_cast<const char*>(input2.raw_data()),
            &(temp_params[1]));
      } else {
        CAFFE_ENFORCE(
            OperatorBase::HasArgument("axis"),
            "Argument `axis` is missing and was not specified as input.");
      }
    } else {
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("tiles"),
          "Argument `tiles` is missing and was not specified as input.");
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("axis"),
          "Argument `axis` is missing and was not specified as input.");
    }

    tiles_ = temp_params[0];
    axis_ = temp_params[1];

    auto* output = Output(0);
    const auto axis = input.canonical_axis_index(axis_);

    // reshape output to be input tiled along the axis
    vector<TIndex> output_dims(input.dims());
    output_dims[axis_] = output_dims[axis_] * tiles_;
    output->Resize(output_dims);

    // size up to (and not including) axis
    const auto outer_dim = input.size_to_dim(axis);
    // size from axis up
    const auto inner_dim = input.size_from_dim(axis);

    /**
     * How this works:
     * Imagine a 2D tensor (matrix) of size 3x10, tiled 2 times.
     * - Tiling along axis 0 (row) means copying the entire 3x10 Matrix 2
     * times. outer_dim = 0, inner_dim = 30.
     * - Tiling along axis 1 (column) means copying each row 2 times, then
     * proceed to the next row, until the end. outer_dim = 3, inner_dim = 10.
     */
    const char* input_data = static_cast<const char*>(input.raw_data());
    char* output_data =
        static_cast<char*>(output->raw_mutable_data(input.meta()));

    DoTile(
        input.meta(),
        input.itemsize(),
        outer_dim,
        inner_dim,
        input_data,
        output_data);

    return true;
  }

 private:
  void DoTile(
      const TypeMeta& meta,
      int item_size,
      int outer_dim,
      int inner_dim,
      const char* input_data,
      char* output_data) {
    for (auto i = 0; i < outer_dim; ++i) {
      for (auto t = 0; t < tiles_; ++t) {
        context_.template CopyItems<Context, Context>(
            meta, inner_dim, input_data, output_data);
        output_data += inner_dim * item_size;
      }
      input_data += inner_dim * item_size;
    }
  }

  int32_t tiles_;
  int32_t axis_;
};

template <typename T, class Context>
class TileGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TileGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        tiles_(OperatorBase::GetSingleArgument<int32_t>("tiles", 1)),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 0)) {}
  ~TileGradientOp() {}

  bool RunOnDevice() override {
    std::array<int32_t, 2> temp_params = {{tiles_, axis_}};
    if (InputSize() > 1) {
      // We potentially have tiles and/or axis specified as inputs
      // as well. We will check for them in that order. In other words:
      // InputSize() == 2: tiles is specified
      // InputSize() == 3: tiles is specified and axis.
      // Anything specified as input will override the arguments
      CAFFE_ENFORCE(
          Input(1).ndim() == 1 && Input(1).size() == 1,
          "Input `tiles` should be a vector of size 1.");

      const auto& input1 = Input(1);
      context_.template CopyItems<Context, CPUContext>(
          input1.meta(),
          1,
          static_cast<const char*>(input1.raw_data()),
          &(temp_params[0]));

      if (InputSize() > 2) {
        CAFFE_ENFORCE(
            Input(2).ndim() == 1 && Input(2).size() == 1,
            "Input `axis` should be a vector of size 1.");

        const auto& input2 = Input(2);
        context_.template CopyItems<Context, CPUContext>(
            input2.meta(),
            1,
            static_cast<const char*>(input2.raw_data()),
            &(temp_params[1]));
      } else {
        CAFFE_ENFORCE(
            OperatorBase::HasArgument("axis"),
            "Argument `axis` is missing and was not specified as input.");
      }
    } else {
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("tiles"),
          "Argument `tiles` is missing and was not specified as input.");
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("axis"),
          "Argument `axis` is missing and was not specified as input.");
    }

    tiles_ = temp_params[0];
    axis_ = temp_params[1];

    const auto& input = Input(0);
    auto* output = Output(0);
    const auto axis = input.canonical_axis_index(axis_);

    // reshape output to be input "untiled" along the axis
    vector<TIndex> output_dims(input.dims());
    output_dims[axis_] = output_dims[axis_] / tiles_;
    output->Resize(output_dims);

    // size up to (and not including) axis
    const auto outer_dim = output->size_to_dim(axis);
    // size from axis up
    const auto inner_dim = output->size_from_dim(axis);

    /**
     * How this works:
     * Imagine a 2D tensor (matrix) of size 3x10, tiled 2 times along axis 1
     * (column).
     * This is equivalent to multiplying by a vector of 1s transposed.
     * The gradient of this is all 1s in the shape of the input matrix
     * (call it X).
     * So the output gradient should be the matrix multipication result
     * of input gradient (gradient of tiled tensor output) and X.
     */
    const char* input_data = static_cast<const char*>(input.raw_data());
    char* output_data =
        static_cast<char*>(output->raw_mutable_data(input.meta()));

    DoTileGradient(
        input.meta(),
        input.itemsize(),
        outer_dim,
        inner_dim,
        input_data,
        output_data);

    return true;
  }

 private:
  void DoTileGradient(
      const TypeMeta& meta,
      int item_size,
      int outer_dim,
      int inner_dim,
      const char* input_data,
      char* output_data) {
    for (auto i = 0; i < outer_dim; ++i) {
      context_.template CopyItems<Context, Context>(
          meta, inner_dim, input_data, output_data);
      input_data += inner_dim * item_size;
      for (auto t = 1; t < tiles_; ++t) {
        math::Axpy<T, Context>(
            inner_dim,
            T(1),
            reinterpret_cast<const T*>(input_data),
            reinterpret_cast<T*>(output_data),
            &context_);
        input_data += inner_dim * item_size;
      }
      output_data += inner_dim * item_size;
    }
  }

  int32_t tiles_;
  int32_t axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TILE_OP_H_
