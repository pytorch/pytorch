#ifndef CAFFE2_OPERATORS_PARTITION_OPS_H_
#define CAFFE2_OPERATORS_PARTITION_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Index>
static inline int moduloPartition(Index key, int numPartitions) {
  int shard = key % numPartitions;
  // equivalent to `if (shard < 0) shard += partitions;`
  shard += numPartitions & (shard >> (sizeof(int) * 8 - 1));
  return shard;
}

class GatherByKeyOp : public Operator<CPUContext> {
 public:
  USE_DISPATCH_HELPER;
  USE_OPERATOR_FUNCTIONS(CPUContext);
  GatherByKeyOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

 private:
  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(0));
  }

 private:
  template <typename Index>
  bool DoRunWithType() {
    const auto numPartitions = InputSize() - 1;
    CAFFE_ENFORCE_GE(numPartitions, 1);
    const auto& keysTensor = Input(0);
    const auto* keysData = keysTensor.template data<Index>();
    const auto& keysShape = Input(0).dims();
    CAFFE_ENFORCE_EQ(
        keysShape.size(), 1, "Only 1D keys tensor supported currently.");

    // 1. Shape and type consistency checks
    const auto& in0Shape = Input(1).dims();
    CAFFE_ENFORCE_GE(in0Shape.size(), 1);

    vector<TIndex> outShape(keysShape);
    outShape.insert(outShape.end(), in0Shape.begin() + 1, in0Shape.end());

    CAFFE_ENFORCE_GE(outShape.size(), 1);
    auto totalSize = in0Shape[0];
    auto meta = Input(1).meta();
    for (int i = 2; i < InputSize(); ++i) {
      const auto& input = Input(i);
      CAFFE_ENFORCE(meta == input.meta());
      CAFFE_ENFORCE_GE(input.ndim(), 1);
      CAFFE_ENFORCE(std::equal(
          outShape.begin() + keysShape.size(),
          outShape.end(),
          input.dims().begin() + 1));
      totalSize += input.dim(0);
    }
    CAFFE_ENFORCE_EQ(keysTensor.size(), totalSize);

    auto* outTensor = Output(0);
    outTensor->Resize(outShape);
    auto* outData = static_cast<char*>(outTensor->raw_mutable_data(meta));
    const auto blockSize = outTensor->size_from_dim(1);

    inputDatas_.resize(numPartitions);
    for (int i = 0; i < numPartitions; ++i) {
      inputDatas_[i] = static_cast<const char*>(Input(i + 1).raw_data());
    }
    inStartOffsets_.assign(numPartitions, 0);
    Index outStartOffset = 0;
    int currentShard = -1;

    // 2. copy from inputs into output based on shard for each input key
    const auto numEntries = keysTensor.size();
    for (int64_t i = 0; i <= numEntries; ++i) {
      auto newShard =
          i < numEntries ? moduloPartition(keysData[i], numPartitions) : -1;
      if (newShard != currentShard) {
        if (currentShard != -1) {
          auto inStartOffset = inStartOffsets_[currentShard];
          auto numItems = i - outStartOffset;
          context_.template CopyItems<CPUContext, CPUContext>(
              meta,
              numItems * blockSize,
              inputDatas_[currentShard] +
                  inStartOffset * blockSize * meta.itemsize(),
              outData + outStartOffset * blockSize * meta.itemsize());
          inStartOffsets_[currentShard] += numItems;
        }
        currentShard = newShard;
        outStartOffset = i;
      }
    }

    return true;
  }

  std::vector<const char*> inputDatas_;
  std::vector<int64_t> inStartOffsets_;
};

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
      int shard = moduloPartition(data[p], partitions);
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
      int shard = moduloPartition(data[p], partitions);
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
        int shard = moduloPartition(data[index], partitions);
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
