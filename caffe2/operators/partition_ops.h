#ifndef CAFFE2_OPERATORS_PARTITION_OPS_H_
#define CAFFE2_OPERATORS_PARTITION_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class ShardingOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  ShardingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "pack_first_input", pack_first_input_, 0) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

 private:
  template <typename Index>
  bool DoRunWithType() {
    CHECK_EQ(OutputSize() % InputSize(), 0)
        << "Output number must be a multiple of input number";
    int partitions = OutputSize() / InputSize();
    CHECK_GT(partitions, 0);

    auto& main_input = Input(0);
    TIndex size = main_input.size();
    const Index* data = main_input.template data<Index>();
    counts_.assign(partitions, 0);
    for (TIndex p = 0; p < size; p++) {
      // TODO: support other partition functions
      int shard = data[p] % partitions;
      // equivalent to `if (shard < 0) shard += partitions;`
      shard += partitions & (shard >> (sizeof(int) * 8 - 1));
      ++counts_[shard];
    }

    raw_datas_.resize(InputSize());
    block_sizes_.resize(InputSize());
    out_datas_.resize(OutputSize());
    for (int i = 0; i < InputSize(); ++i) {
      auto& input = Input(i);
      if (i > 0) {
        CHECK_GE(input.ndim(), main_input.ndim())
            << "Prefix of extra input's shape must match main input's shape, "
            << "input: " << i;
        for (int j = 0; j < main_input.ndim(); ++j) {
          CHECK_GE(input.dim(j), main_input.dim(j))
              << "Prefix of extra input's shape must match main input's shape, "
              << "input: " << i << ", dim " << j;
        }
        CHECK(input.meta().copy() == nullptr)
            << "Only primitive types are supported, input " << i;
      }
      raw_datas_[i] = input.raw_data();
      block_sizes_[i] =
          input.size_from_dim(main_input.ndim()) * input.itemsize();
      // shape = partition_size + suffix of input dims
      vector<TIndex> shape(
          input.dims().begin() + main_input.ndim() - 1, input.dims().end());
      for (int j = 0; j < partitions; ++j) {
        int out_idx = i * partitions + j;
        auto* output = Output(out_idx);
        shape[0] = counts_[j];
        output->Resize(shape);
        out_datas_[out_idx] = output->raw_mutable_data(input.meta());
      }
    }

    counts_.assign(partitions, 0);
    for (TIndex p = 0; p < size; p++) {
      // TODO: support other partition functions
      int shard = data[p] % partitions;
      // equivalent to `if (shard < 0) shard += partitions;`
      shard += partitions & (shard >> (sizeof(int) * 8 - 1));
      TIndex idx = counts_[shard]++;

      // special case first input
      static_cast<Index*>(out_datas_[shard])[idx] =
          pack_first_input_ ? ((data[p] - shard) / partitions) : data[p];

      for (int i = 1, j = shard + partitions; i < InputSize();
           ++i, j += partitions) {
        auto bs = block_sizes_[i];
        // special case for small bs?
        context_.template CopyBytes<Context, Context>(
            bs,
            static_cast<const char*>(raw_datas_[i]) + p * bs,
            static_cast<char*>(out_datas_[j]) + idx * bs);
      }
    }

    return true;
  }

  bool pack_first_input_;

  // use member fields to reuse memory
  vector<TIndex> counts_;
  vector<TIndex> block_sizes_;
  vector<const void*> raw_datas_;
  vector<void*> out_datas_;

  DISABLE_COPY_AND_ASSIGN(ShardingOp);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PARTITION_OPS_H_
