#ifndef CAFFE2_OPERATORS_LENGTHS_TILE_OP_H_
#define CAFFE2_OPERATORS_LENGTHS_TILE_OP_H_

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class LengthsTileOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(LengthsTileOp);

  bool RunOnDevice() override {
    auto& data = Input(DATA);
    auto& lengths = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(lengths.ndim(), 1, "LENGTHS must be 1-D");
    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
    CAFFE_ENFORCE_EQ(lengths.size(), data.dim(0));

    auto* lengths_data = lengths.template data<int32_t>();
    int32_t total_length = 0;
    math::Sum<int32_t, Context>(
        lengths.size(), lengths_data, &total_length, &context_);

    auto shape = data.dims();
    shape[0] = total_length;
    output->Resize(shape);

    auto block_size = data.size_from_dim(1);
    auto block_bytesize = data.size_from_dim(1) * data.meta().itemsize();
    auto N = lengths.size();

    auto src = static_cast<const char*>(data.raw_data());
    auto out = static_cast<char*>(output->raw_mutable_data(data.meta()));

    for (TIndex i = 0; i < N; ++i) {
      auto length = lengths_data[i];
      CAFFE_ENFORCE_GE(length, 0);
      for (int32_t j = 0; j < length; ++j) {
        context_.template CopyItems<Context, Context>(
            data.meta(), block_size, src, out);
        out += block_bytesize;
      }
      src += block_bytesize;
    }
    return true;
  }

  INPUT_TAGS(DATA, LENGTHS);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_TILE_OP_H_
