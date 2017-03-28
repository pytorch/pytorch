#ifndef CAFFE2_OPERATORS_PARTITION_OPS_H_
#define CAFFE2_OPERATORS_PARTITION_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

class PartitionOpBase : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);

  PartitionOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        OP_SINGLE_ARG(int, "pack_first_input", pack_first_input_, 0) {}

 protected:
  template <typename Index>
  void ApplyPartition(bool skipFirstArgument) {
    CAFFE_ENFORCE_EQ(
        OutputSize() % InputSize(),
        0,
        "Output number must be a multiple of input number");
    int partitions = OutputSize() / InputSize();
    int inputSize = InputSize();
    int mainInputIndex = skipFirstArgument;
    CAFFE_ENFORCE_GT(partitions, 0, "Invalid number of partitions");

    auto& main_input = Input(mainInputIndex);
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

    raw_datas_.resize(inputSize);
    block_sizes_.resize(inputSize);
    metas_.resize(inputSize);
    out_datas_.resize(OutputSize());
    for (int i = mainInputIndex; i < inputSize; ++i) {
      auto& input = Input(i);
      if (i > mainInputIndex) {
        CAFFE_ENFORCE_GE(
            input.ndim(),
            main_input.ndim(),
            "Prefix of extra input's shape must match main input's shape, ",
            "input: ",
            i);
        for (int j = 0; j < main_input.ndim(); ++j) {
          CAFFE_ENFORCE_GE(
              input.dim(j),
              main_input.dim(j),
              "Prefix of extra input's shape must match main input's shape, ",
              "input: ",
              i,
              ", dim ",
              j);
        }
      }
      raw_datas_[i] = input.raw_data();
      block_sizes_[i] = input.size_from_dim(main_input.ndim());
      metas_[i] = input.meta();
      // shape = partition_size + suffix of input dims
      vector<TIndex> shape(
          input.dims().begin() + main_input.ndim() - 1, input.dims().end());
      for (int j = 0; j < partitions; ++j) {
        int out_idx = i + j * inputSize;
        auto output = Output(out_idx);
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
      static_cast<Index*>(out_datas_[shard * inputSize + mainInputIndex])[idx] =
          pack_first_input_ ? ((data[p] - shard) / partitions) : data[p];

      int baseIndex = shard * inputSize;
      for (int i = mainInputIndex + 1; i < inputSize; ++i) {
        auto bs = block_sizes_[i];
        auto meta = metas_[i];
        // special case for small bs?
        context_.template CopyItems<CPUContext, CPUContext>(
            meta,
            bs,
            static_cast<const char*>(raw_datas_[i]) + p * bs * meta.itemsize(),
            static_cast<char*>(out_datas_[baseIndex + i]) +
                idx * bs * meta.itemsize());
      }
    }
  }

  bool pack_first_input_;

  // use member fields to reuse memory
  vector<TIndex> counts_;
  vector<TIndex> block_sizes_;
  vector<TypeMeta> metas_;
  vector<const void*> raw_datas_;
  vector<void*> out_datas_;
};

class PartitionOp : public PartitionOpBase {
 public:
  USE_DISPATCH_HELPER;

  PartitionOp(const OperatorDef& operator_def, Workspace* ws)
      : PartitionOpBase(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

 private:
  template <typename Index>
  bool DoRunWithType() {
    ApplyPartition<Index>(false /* skipFirstArgument */);
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(PartitionOp);
};

class LengthsPartitionOp : public PartitionOpBase {
 public:
  USE_DISPATCH_HELPER;

  LengthsPartitionOp(const OperatorDef& operator_def, Workspace* ws)
      : PartitionOpBase(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(1));
  }

 private:
  template <typename Index>
  bool DoRunWithType() {
    CAFFE_ENFORCE(
        OutputSize() % InputSize() == 0,
        "Output number must be a multiple of input number");
    int partitions = OutputSize() / InputSize();
    CAFFE_ENFORCE_GT(partitions, 0, "Invalid number of partitions");
    CAFFE_ENFORCE_EQ(
        Input(1).ndim(),
        1,
        "Only 1-D tensors supported as a partitioning tensor for sharding");

    // Apply sharding to all parameters except lengths
    ApplyPartition<Index>(true /* skipFirstArgument */);

    // Compute lengths after sharding
    auto& main_input = Input(1);
    TIndex size = main_input.size();
    const Index* data = main_input.template data<Index>();

    auto& length_input = Input(0);
    TIndex elements = length_input.size();
    const int32_t* lengths_data = length_input.template data<int32_t>();
    out_length_.resize(partitions);
    for (int i = 0; i < partitions; ++i) {
      auto& output = *Output(i * InputSize());
      output.Resize(elements);
      out_length_[i] = output.template mutable_data<int32_t>();
    }

    int total_length = 0;
    for (int i = 0; i < elements; ++i) {
      total_length += lengths_data[i];
    }
    CAFFE_ENFORCE(
        total_length == size,
        "Total length is not matching to the number of elements");

    int index = 0;
    for (int i = 0; i < elements; ++i) {
      for (int j = 0; j < partitions; ++j) {
        out_length_[j][i] = 0;
      }
      for (int j = 0; j < lengths_data[i]; ++j, ++index) {
        // TODO: support other partition functions
        int shard = data[index] % partitions;
        // equivalent to `if (shard < 0) shard += partitions;`
        shard += partitions & (shard >> (sizeof(int) * 8 - 1));
        ++out_length_[shard][i];
      }
    }
    return true;
  }

  DISABLE_COPY_AND_ASSIGN(LengthsPartitionOp);

  vector<int32_t*> out_length_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PARTITION_OPS_H_
