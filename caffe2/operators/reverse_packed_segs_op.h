#ifndef CAFFE2_OPERATORS_REVERSE_PACKED_SEGS_OP_H_
#define CAFFE2_OPERATORS_REVERSE_PACKED_SEGS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "c10/util/irange.h"

namespace caffe2 {

template <class Context>
class ReversePackedSegsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ReversePackedSegsOp);
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double, int, long, bool>>::call(
        this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType() {
    if (Input(LENGTHS).template IsType<int>()) {
      DoRunWithLengthType<T, int>();
    } else {
      DoRunWithLengthType<T, long>();
    }
    return true;
  }

 private:
  INPUT_TAGS(DATA, LENGTHS);

  template <typename T, typename LengthType>
  void DoRunWithLengthType() {
    const auto& data = Input(DATA);
    const auto& lengths = Input(LENGTHS);

    CAFFE_ENFORCE(
        data.dim() == 3,
        "DATA should be 3-D tensor <lengths, "
        "segments, embeddings>");
    CAFFE_ENFORCE(lengths.dim() == 1, "LENGTH should be 1-D");

    const auto shape = data.sizes();
    auto* output = Output(0, shape, at::dtype<T>());

    const auto max_length = data.sizes()[0];
    const auto batch_size = data.sizes()[1];
    const auto block_size = data.sizes()[2];
    CAFFE_ENFORCE(
        lengths.sizes()[0] == batch_size,
        "lenths size should be"
        " equal to batch size");

    const T* data_ptr = data.template data<T>();
    const LengthType* lengths_ptr = lengths.template data<LengthType>();

    vector<LengthType> lengths_host(batch_size);
    context_.template CopyToCPU<LengthType>(
        batch_size, lengths_ptr, &lengths_host[0]);
    context_.FinishDeviceComputation();

    T* rev_data_ptr = output->template mutable_data<T>();
    for (const auto i : c10::irange(batch_size)) {
      const auto& seg_length = lengths_host[i];
      CAFFE_ENFORCE_LE(seg_length, max_length);
      int64_t j = 0;
      for (; j < seg_length; j++) {
        const T* data_block_ptr = data_ptr + (j * batch_size + i) * block_size;
        T* rev_data_block_ptr =
            rev_data_ptr + ((seg_length - 1 - j) * batch_size + i) * block_size;
        context_.template CopySameDevice<T>(
            block_size, data_block_ptr, rev_data_block_ptr);
      }
      for (; j < max_length; j++) {
        const T* data_block_ptr = data_ptr + (j * batch_size + i) * block_size;
        T* rev_data_block_ptr =
            rev_data_ptr + (j * batch_size + i) * block_size;
        context_.template CopySameDevice<T>(
            block_size, data_block_ptr, rev_data_block_ptr);
      }
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REVERSE_PACKED_SEGS_OP_H_
