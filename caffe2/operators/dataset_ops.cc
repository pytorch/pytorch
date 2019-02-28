#include "caffe2/operators/dataset_ops.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(std::unique_ptr<dataset_ops::TreeCursor>);
CAFFE_KNOWN_TYPE(dataset_ops::TensorVectorPtr);
CAFFE_KNOWN_TYPE(dataset_ops::SharedTensorVectorPtr);

namespace dataset_ops {
namespace {

const char kDatasetFieldSeparator = ':';
const char* kDatasetLengthField = "lengths";

// how much percent to grow the dataset when needed
const int kDatasetGrowthPct = 40;

} // namespace

TreeIterator::TreeIterator(const std::vector<std::string>& fields) {
  // populate field vector and split field names
  fields_.resize(fields.size());
  std::vector<std::vector<std::string>> nameParts(fields_.size());
  for (int i = 0; i < fields.size(); ++i) {
    auto& field = fields_.at(i);
    field.name = fields[i];
    field.id = i;
    field.lengthFieldId = -1;
    nameParts.at(i) = split(kDatasetFieldSeparator, field.name);
  }

  // populate lengthFields
  for (const auto& field : fields_) {
    const auto& parts = nameParts.at(field.id);
    if (!parts.empty() && parts.back() == kDatasetLengthField) {
      lengthFieldIds_.push_back(field.id);
    }
  }

  // find length-field with maximum prefix matching for each field
  for (auto& field : fields_) {
    // by default, we are matching against the root domain
    int maxMatchLevel = 1;
    int maxMatchLengthFieldId = -1;
    for (int j = 0; j < numLengthFields(); ++j) {
      const auto& lenField = lengthField(j);
      // a length field can't have itself as its length field
      if (field.id == lenField.id) {
        continue;
      }
      auto lf = nameParts.at(lenField.id);
      auto lfEnd = lf.end() - 1;
      // check whether this lengthField is a prefix for this field name
      if (std::mismatch(lf.begin(), lfEnd, nameParts.at(field.id).begin())
              .first != lfEnd) {
        continue;
      }
      if (lf.size() > maxMatchLevel) {
        maxMatchLevel = lf.size();
        maxMatchLengthFieldId = j;
      }
    }
    field.lengthFieldId = maxMatchLengthFieldId;
  }

  // check that fields are topologically sorted
  // (no length field depends on a length defined afterwards)
  for (const auto& field : fields_) {
    const auto* lengthField = lengthFieldFor(field);
    CAFFE_ENFORCE(
        (lengthField == nullptr) || (lengthField->id < field.id),
        "Error: Field ",
        field.id,
        " (",
        field.name,
        ") ",
        "depends on a field defined afterwards: ",
        lengthField->id,
        " (",
        lengthField->name,
        ").");
  }
}

void TreeIterator::advance(
    const std::vector<const TLength*>& lengths,
    std::vector<TOffset>& offsets,
    std::vector<TOffset>& sizes,
    std::vector<TOffset>& limits,
    TOffset num) {
  std::vector<TOffset> newOffsets;
  CAFFE_ENFORCE_EQ(lengths.size(), numLengthFields());
  CAFFE_ENFORCE_EQ(offsets.size(), numOffsetFields());
  sizes.resize(offsets.size());
  newOffsets.resize(offsets.size());
  // first index, top level
  {
    auto limit = limits[0];
    auto offset = offsets[0];
    CAFFE_ENFORCE(limit >= offset, "Tried to advance past end of cursor.");
    TOffset total = std::min(limit - offset, num);
    sizes[0] = total;
    newOffsets[0] = offset + total;
  }
  // child indices
  for (int j = 1; j < numOffsetFields(); ++j) {
    TOffset total = 0;
    int parentOffsetId = offsetFieldIdFor(lengthField(j - 1));
    const TLength* length = lengths[j - 1] + offsets[parentOffsetId];
    for (int k = 0; k < sizes[parentOffsetId]; ++k) {
      total += *(length++);
    }
    auto offset = offsets[j];
    CAFFE_ENFORCE(
        offset + total <= limits[j],
        "Inconsistent field length: ",
        "tried to advance past the end of field ",
        j);
    sizes[j] = total;
    newOffsets[j] = offset + total;
  }
  offsets = newOffsets;
}

TreeWalker::TreeWalker(const vector<const Blob*>& inputs, TreeCursor& cursor)
    : inputs_(inputs), cursor_(cursor), sizes_(cursor.it.numOffsetFields()) {
  CAFFE_ENFORCE_EQ(inputs.size(), cursor.it.fields().size());
  if (cursor.offsets.empty()) {
    cursor.offsets.assign(cursor.it.numOffsetFields(), 0);
  }

  for (int fieldId = 0; fieldId < cursor_.it.fields().size(); ++fieldId) {
    fields_.emplace_back(*this, fieldId);
  }

  gatherLengthData();

  gatherSizeLimits();

  // The invariant we hold is that we are always one step ahead
  advance();
}

void TreeWalker::advance() {
  prevOffsets_ = cursor_.offsets;
  cursor_.it.advance(lengths_, cursor_.offsets, sizes_, limits_, 1);
}

std::vector<int64_t> TreeWalker::fieldDim(int fieldId) const {
  auto tensorDim = input(fieldId).sizes().vec();
  tensorDim[0] = sizes_[lengthIdx(fieldId)];
  return tensorDim;
}

void* TreeWalker::fieldPtr(int fieldId) const {
  auto& in = input(fieldId);
  return (char*)in.raw_data() +
      offset(fieldId) * in.size_from_dim(1) * in.dtype().itemsize();
}

void TreeWalker::gatherLengthData() {
  static const TLength lenZero = 0;
  lengths_.resize(cursor_.it.numLengthFields());
  for (int i = 0; i < lengths_.size(); ++i) {
    auto& in = input(cursor_.it.lengthField(i).id);
    if (in.numel() > 0) {
      lengths_[i] = in.data<int>();
    } else {
      lengths_[i] = &lenZero;
    }
  }
}

void TreeWalker::gatherSizeLimits() {
  limits_.assign(sizes_.size(), std::numeric_limits<TOffset>::max());
  for (auto fieldId = 0; fieldId < cursor_.it.fields().size(); ++fieldId) {
    auto lengthFieldIdx = lengthIdx(fieldId);
    limits_[lengthFieldIdx] =
        std::min(limits_[lengthFieldIdx], (TOffset)input(fieldId).sizes()[0]);
  }
}

namespace {

class CreateTreeCursorOp : public Operator<CPUContext> {
 public:
  CreateTreeCursorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        fields_(OperatorBase::GetRepeatedArgument<std::string>("fields")) {}

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<TreeCursor>>(0) =
        std::unique_ptr<TreeCursor>(new TreeCursor(TreeIterator(fields_)));
    return true;
  }

 private:
  std::vector<std::string> fields_;
};

class GetCursorOffsetOp : public Operator<CPUContext> {
 public:
  GetCursorOffsetOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& cursor = OperatorBase::Input<std::unique_ptr<TreeCursor>>(0);
    Output(0)->Resize(cursor->offsets.size());
    auto* output = Output(0)->template mutable_data<int>();
    for (size_t i = 0; i < cursor->offsets.size(); ++i) {
      output[i] = cursor->offsets[i];
    }
    return true;
  }
};

class ResetCursorOp : public Operator<CPUContext> {
 public:
  ResetCursorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& cursor = OperatorBase::Input<std::unique_ptr<TreeCursor>>(0);
    std::lock_guard<std::mutex> lock(cursor->mutex_);
    cursor->offsets.clear();
    return true;
  }
};

class CheckDatasetConsistencyOp : public Operator<CPUContext> {
 public:
  CheckDatasetConsistencyOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        iterator_(OperatorBase::GetRepeatedArgument<std::string>("fields")) {}

  bool RunOnDevice() override {
    std::vector<const TLength*> lengths;
    std::vector<TOffset> limits;
    std::vector<TOffset> sizes;
    std::vector<TOffset> offsets;
    CAFFE_ENFORCE(
        InputSize() == iterator_.fields().size(),
        "Invalid number of fields. Expected ",
        iterator_.fields().size(),
        ", got ",
        InputSize());
    sizes.resize(iterator_.numOffsetFields());
    // gather length data
    lengths.resize(iterator_.numLengthFields());
    for (int i = 0; i < lengths.size(); ++i) {
      lengths[i] = Input(iterator_.lengthField(i).id).data<TLength>();
    }
    // gather size limits
    limits.assign(sizes.size(), std::numeric_limits<TOffset>::max());
    for (int i = 0; i < iterator_.fields().size(); ++i) {
      int lengthIdx = iterator_.fields()[i].lengthFieldId + 1;
      CAFFE_ENFORCE_GT(Input(i).dim(), 0);
      TOffset size = (TOffset)Input(i).sizes()[0];
      if (limits[lengthIdx] == std::numeric_limits<TOffset>::max()) {
        limits[lengthIdx] = size;
      } else {
        CAFFE_ENFORCE(
            limits[lengthIdx] == size,
            "Inconsistent sizes for fields belonging to same domain.",
            " Field: ",
            i,
            " (",
            iterator_.fields()[i].name,
            "); Length field index: ",
            lengthIdx,
            "); Previous size: ",
            limits[lengthIdx],
            "; New size: ",
            size);
      }
    }
    // advance to the end
    offsets.assign(sizes.size(), 0);
    iterator_.advance(lengths, offsets, sizes, limits, limits[0]);
    for (int i = 0; i < limits.size(); ++i) {
      CAFFE_ENFORCE(limits[i] == offsets[i]);
    }
    return true;
  }

 private:
  TreeIterator iterator_;
};

class PackRecordsOp : public Operator<CPUContext> {
 public:
  PackRecordsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        fields_(OperatorBase::GetRepeatedArgument<std::string>("fields")) {}

  bool RunOnDevice() override {
    // There should be one input per field
    CAFFE_ENFORCE_EQ(InputSize(), fields_.size());
    CAFFE_ENFORCE_EQ(OutputSize(), 1);

    TreeCursor cursor((TreeIterator(fields_)));

    TreeWalker walker(Inputs(), cursor);

    Output(0)->Resize(walker.size());

    // Output(0)->raw_mutable_data(TypeMeta::Make<SharedTensorVectorPtr>()));
    auto* dst = Output(0)->template mutable_data<SharedTensorVectorPtr>();

    for (int batchId = 0; batchId < walker.size(); ++batchId) {
      dst[batchId] = std::make_shared<std::vector<TensorCPU>>();
      dst[batchId]->reserve(walker.fields().size());

      for (const auto& field : walker.fields()) {
        dst[batchId]->emplace_back(field.dim(), CPU);
        auto& tensor = dst[batchId]->back();
        context_.CopyItemsSameDevice(
            field.meta(),
            tensor.numel(),
            field.ptr() /* src */,
            tensor.raw_mutable_data(field.meta()) /* dst */);
      }

      walker.advance();
    }

    return true;
  }

 private:
  std::vector<std::string> fields_;
};

class UnPackRecordsOp : public Operator<CPUContext> {
 public:
  UnPackRecordsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        fields_(OperatorBase::GetRepeatedArgument<std::string>("fields")) {}

  bool RunOnDevice() override {
    const auto* inputs = Input(0).template data<SharedTensorVectorPtr>();
    const auto numRows = Input(0).numel();

    CAFFE_ENFORCE_GE(numRows, 0);

    auto numTensors = OutputSize();

    // Precomputer the output sizes to avoid resizing
    std::vector<std::vector<int64_t>> outputDims(numTensors);
    std::vector<const TypeMeta*> metas(numTensors);

    CAFFE_ENFORCE(
        numRows > 0 || InputSize() > 1,
        "Unpacking empty record without shape will leave output blobs in "
        "undefined state.");

    if (InputSize() == 1) {
      getShapeAndMetaFromInput(outputDims, metas);
    } else {
      getShapeAndMetaFromPrototypeBlobs(outputDims, metas);
    }

    for (int i = 0; i < numRows; ++i) {
      CAFFE_ENFORCE(inputs[i]);
      for (int j = 0; j < inputs[i]->size(); ++j) {
        const auto& input = inputs[i]->at(j);

        // Checks to ensure that dimensions/sizes match
        CAFFE_ENFORCE_EQ(outputDims[j].size(), input.dim());
        CAFFE_ENFORCE(*metas[j] == input.dtype());
        // We look from first dimension, because we concat on the first.
        for (int k = 1; k < input.dim(); ++k) {
          CAFFE_ENFORCE_EQ(input.sizes()[k], outputDims[j][k]);
        }

        outputDims[j][0] += input.size(0);
      }
    }

    // Resize to the final output size
    std::vector<void*> destinations(numTensors);
    for (int i = 0; i < numTensors; ++i) {
      Output(i)->Resize(outputDims[i]);
      destinations[i] = Output(i)->raw_mutable_data(*metas[i]);
    }

    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numTensors; ++j) {
        const auto& input = inputs[i]->at(j);

        context_.CopyItemsSameDevice(
            *metas[j],
            input.numel(),
            input.raw_data() /* src */,
            destinations[j] /* dst */
        );

        destinations[j] =
            (char*)destinations[j] + input.numel() * input.itemsize();
      }
    }

    return true;
  }

 private:
  void getShapeAndMetaFromInput(
      std::vector<std::vector<int64_t>>& outputDims,
      std::vector<const TypeMeta*>& metas) {
    const auto* inputs = Input(0).template data<SharedTensorVectorPtr>();

    const auto& inputZero = inputs[0];
    CAFFE_ENFORCE(inputZero);

    const auto numTensors = inputZero->size();

    CAFFE_ENFORCE_EQ(numTensors, fields_.size());
    CAFFE_ENFORCE_EQ(numTensors, OutputSize());

    for (int i = 0; i < numTensors; ++i) {
      outputDims[i] = inputZero->at(i).sizes().vec();
      outputDims[i][0] = 0;
      metas[i] = &inputZero->at(i).dtype();
    }
  }

  void getShapeAndMetaFromPrototypeBlobs(
      std::vector<std::vector<int64_t>>& outputDims,
      std::vector<const TypeMeta*>& metas) {
    const auto numTensors = fields_.size();
    CAFFE_ENFORCE_EQ(numTensors, InputSize() - 1);
    CAFFE_ENFORCE_EQ(numTensors, OutputSize());
    for (int i = 0; i < numTensors; ++i) {
      const auto& input = Input(i + 1);
      outputDims[i] = input.sizes().vec();
      outputDims[i][0] = 0;
      metas[i] = &input.dtype();
    }
  }

  std::vector<std::string> fields_;
};

class ReadNextBatchOp : public Operator<CPUContext> {
 public:
  ReadNextBatchOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        batchSize_(OperatorBase::GetSingleArgument<int>("batch_size", 1)),
        enforceBatchSize_(OperatorBase::GetSingleArgument<bool>(
            "enforce_batch_size",
            false)) {}

  bool RunOnDevice() override {
    auto& cursor = OperatorBase::Input<std::unique_ptr<TreeCursor>>(0);
    CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
    std::vector<const TLength*> lengths;
    std::vector<TOffset> limits;
    std::vector<TOffset> sizes;
    std::vector<TOffset> offsets;
    TLength lenZero = 0;
    sizes.resize(cursor->it.numOffsetFields());
    // gather length data
    lengths.resize(cursor->it.numLengthFields());
    for (int i = 0; i < lengths.size(); ++i) {
      auto& a = Input(cursor->it.lengthField(i).id + 1);
      if (a.numel() > 0) {
        lengths[i] = a.data<int>();
      } else {
        lengths[i] = &lenZero;
      }
    }
    // gather size limits
    limits.assign(sizes.size(), std::numeric_limits<TOffset>::max());
    for (int i = 0; i < cursor->it.fields().size(); ++i) {
      int lengthFieldIdx = cursor->it.fields()[i].lengthFieldId + 1;
      limits[lengthFieldIdx] =
          std::min(limits[lengthFieldIdx], (TOffset)Input(i + 1).sizes()[0]);
    }
    // advance cursor
    {
      std::lock_guard<std::mutex> lock(cursor->mutex_);
      if (cursor->offsets.empty()) {
        cursor->offsets.assign(sizes.size(), 0);
      }
      offsets = cursor->offsets;
      cursor->it.advance(lengths, cursor->offsets, sizes, limits, batchSize_);
      if (enforceBatchSize_ && sizes[0] < batchSize_) {
        // if we enforce batch_size but don't have enough rows left to
        // complete a full batch, return empty for all columns.
        // This signals end of dataset to the caller.
        sizes.assign(sizes.size(), 0);
      }
    }
    // gather data
    std::vector<int64_t> outDim;
    for (int i = 0; i < cursor->it.fields().size(); ++i) {
      auto lengthIdx = cursor->it.fields()[i].lengthFieldId + 1;
      auto size = sizes[lengthIdx];
      auto offset = offsets[lengthIdx];
      auto& in = Input(i + 1);
      auto innerSize = in.size_from_dim(1);
      outDim = in.sizes().vec();
      outDim[0] = size;
      auto* out = Output(i);
      out->Resize(outDim);
      void* src =
          (char*)in.raw_data() + offset * innerSize * in.dtype().itemsize();
      void* dst = out->raw_mutable_data(in.dtype()); // create the tensor
      if (out->numel() == 0) {
        continue;
      }
      context_.CopyItemsSameDevice(in.dtype(), out->numel(), src, dst);
    }
    return true;
  }
  int batchSize_;
  bool enforceBatchSize_;
};

class ComputeOffsetOp : public Operator<CPUContext> {
 public:
  ComputeOffsetOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& cursor = OperatorBase::Input<std::unique_ptr<TreeCursor>>(0);
    CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
    auto* out = Output(0);
    std::vector<const TLength*> lengths;
    std::vector<TOffset> limits;
    std::vector<TOffset> sizes;
    std::vector<TOffset> offsets;
    TLength lenZero = 0;
    sizes.resize(cursor->it.numOffsetFields());
    // gather length data
    lengths.resize(cursor->it.numLengthFields());
    for (int i = 0; i < lengths.size(); ++i) {
      auto& a = Input(cursor->it.lengthField(i).id + 1);
      if (a.numel() > 0) {
        lengths[i] = a.data<int>();
      } else {
        lengths[i] = &lenZero;
      }
    }
    // gather size limits
    limits.assign(sizes.size(), std::numeric_limits<TOffset>::max());
    for (int i = 0; i < cursor->it.fields().size(); ++i) {
      int lengthFieldIdx = cursor->it.fields()[i].lengthFieldId + 1;
      limits[lengthFieldIdx] =
          std::min(limits[lengthFieldIdx], (TOffset)Input(i + 1).sizes()[0]);
    }
    out->Resize(limits.at(0) + 1, sizes.size());
    auto* out_data = out->template mutable_data<int64_t>();
    for (int k = 0; k <= limits.at(0); k++) {
      // advance cursor
      if (cursor->offsets.empty()) {
        cursor->offsets.assign(sizes.size(), 0);
      }
      // write output
      std::copy(cursor->offsets.begin(), cursor->offsets.end(), out_data);
      out_data += sizes.size();
      cursor->it.advance(lengths, cursor->offsets, sizes, limits, 1);
    }
    cursor->offsets.assign(sizes.size(), 0); // reSet after getting meta info
    return true;
  }
};

class SortAndShuffleOp : public Operator<CPUContext> {
 public:
  SortAndShuffleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        sort_by_field_idx_(
            OperatorBase::GetSingleArgument<int>("sort_by_field_idx", 1)),
        batch_size_(OperatorBase::GetSingleArgument<int>("batch_size", 1)),
        shuffle_size_(OperatorBase::GetSingleArgument<int>("shuffle_size", 1)) {
  }

  bool RunOnDevice() override {
    auto& cursor = OperatorBase::Input<std::unique_ptr<TreeCursor>>(0);
    CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 1);
    CAFFE_ENFORCE(-1 <= sort_by_field_idx_);
    CAFFE_ENFORCE(cursor->it.fields().size() - sort_by_field_idx_ > 0);
    int size;
    if (sort_by_field_idx_ != -1) {
      size = Input(sort_by_field_idx_ + 1).sizes()[0];
    } else {
      size = Input(1).sizes()[0];
    }

    CAFFE_ENFORCE(
        batch_size_ > 0 && shuffle_size_ > 0 &&
        0 < batch_size_ * shuffle_size_);
    // adjust shuffle_size_ if it is too large
    if (batch_size_ * shuffle_size_ > size) {
      shuffle_size_ = size / batch_size_;
    }

    int num_batch = size / batch_size_;
    auto* out = Output(0);
    out->Resize(size);
    auto* out_data = out->template mutable_data<int64_t>();

    vector<int> shuffle_idx(size);
    iota(shuffle_idx.begin(), shuffle_idx.end(), 0);

    if (sort_by_field_idx_ != -1) {
      auto& sortblob = Input(sort_by_field_idx_ + 1);
      auto* sortdata = sortblob.data<int>();
      // must sort by a field at the root level
      CAFFE_ENFORCE(
          cursor->it.fields()[sort_by_field_idx_].lengthFieldId == -1);
      sort(shuffle_idx.begin(), shuffle_idx.end(), [&sortdata](int i1, int i2) {
        return sortdata[i1] < sortdata[i2];
      });
    }

    if (batch_size_ * shuffle_size_ > 1) {
      int offset = 0;
      while (offset + batch_size_ * shuffle_size_ < size) {
        std::shuffle(
            shuffle_idx.begin() + offset,
            shuffle_idx.begin() + offset + batch_size_ * shuffle_size_,
            std::default_random_engine());
        offset += batch_size_ * shuffle_size_;
      }
    }

    vector<int> batch_idx(num_batch);
    iota(batch_idx.begin(), batch_idx.end(), 0);
    std::shuffle(
        batch_idx.begin(), batch_idx.end(), std::default_random_engine());

    for (int i = 0; i < num_batch; i++) {
      std::copy(
          shuffle_idx.begin() + batch_idx[i] * batch_size_,
          shuffle_idx.begin() + (batch_idx[i] + 1) * batch_size_,
          out_data);
      out_data += batch_size_;
    }
    std::copy(
        shuffle_idx.begin() + num_batch * batch_size_,
        shuffle_idx.end(),
        out_data);

    return true;
  }

  int sort_by_field_idx_;
  int batch_size_;
  int shuffle_size_;
};

class ReadRandomBatchOp : public Operator<CPUContext> {
 public:
  ReadRandomBatchOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        batchSize_(OperatorBase::GetSingleArgument<int>("batch_size", 1)),
        enforceBatchSize_(
            OperatorBase::GetSingleArgument<bool>("enforce_batch_size", false)),
        loopOver_(OperatorBase::GetSingleArgument<bool>("loop_over", false)) {}
  bool RunOnDevice() override {
    auto& cursor = OperatorBase::Input<std::unique_ptr<TreeCursor>>(0);
    auto& idxblob = Input(1);
    auto& offsetsmat = Input(2);
    CAFFE_ENFORCE(InputSize() == cursor->it.fields().size() + 3);
    auto idxvec = idxblob.template data<int64_t>();
    auto offsetdim = offsetsmat.sizes();
    // gather data
    std::vector<int64_t> outDim;
    int64_t idx;
    {
      std::lock_guard<std::mutex> lock(cursor->mutex_);
      cursor->offsets.resize(1);
      idx = cursor->offsets.at(0);
      // if we want to enforce batch size but we dont have a complete
      // batch, skip the last rows.
      if (enforceBatchSize_ && idx + batchSize_ > idxblob.numel()) {
        idx = idxblob.numel();
      }
      if (loopOver_ && idx >= idxblob.numel()) {
        cursor->offsets.at(0) = 0;
        idx = 0;
      }
      cursor->offsets.at(0) += batchSize_;
    }

    for (int i = 0; i < cursor->it.fields().size(); ++i) {
      auto lengthIdx = cursor->it.fields()[i].lengthFieldId + 1;
      auto& in = Input(i + 3);
      outDim = in.sizes().vec();
      outDim.at(0) = 0;
      auto idxbegin = idx;
      for (int j = 0; j < batchSize_; ++j) {
        if (idx >= idxblob.numel()) {
          break;
        }
        CAFFE_ENFORCE(
            (idxvec[idx] + 1) * offsetdim[1] + lengthIdx < offsetsmat.numel(),
            "Out of bound when trying to get elem from offsetsmat");
        auto offsetptr = offsetsmat.template data<TOffset>() +
            idxvec[idx] * offsetdim[1] + lengthIdx;
        auto offset = *offsetptr;
        auto size = *(offsetptr + offsetdim[1]) - offset;
        outDim.at(0) += size; // accumulate over the batch
        idx++;
      }
      idx = idxbegin; // reSet
      auto* out = Output(i);
      out->Resize(outDim);
      if (out->numel() == 0) {
        continue;
      }
      auto dst = static_cast<char*>(out->raw_mutable_data(in.dtype()));
      int block_size = in.numel() / in.size(0);
      auto block_bytesize = in.size_from_dim(1) * in.dtype().itemsize();
      CAFFE_ENFORCE(
          block_bytesize == in.nbytes() / in.size(0),
          "block_bytesize should be consistent with data dim");
      auto src_base = static_cast<const char*>(in.raw_data());
      int start = 0;
      for (int j = 0; j < batchSize_; ++j) {
        if (idx >= idxblob.numel()) {
          break;
        }
        auto offsetptr = offsetsmat.template data<TOffset>() +
            idxvec[idx] * offsetdim[1] + lengthIdx;
        auto offset = *offsetptr;
        auto size = *(offsetptr + offsetdim[1]) - offset;
        // copy data
        auto src = src_base + offset * block_bytesize;
        context_.CopyItemsSameDevice(
            in.dtype(), size * block_size, src, dst + start * block_bytesize);
        start += size;
        idx++;
      }
      idx = idxbegin; // reSet
    }
    return true;
  }
  int batchSize_;
  bool enforceBatchSize_;
  bool loopOver_;
};

template <class Context>
class AppendOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AppendOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& a = Input(0);
    auto& b = Input(1);
    auto* c = Output(0);
    CAFFE_ENFORCE(b.dim() >= 1);
    if (a.numel() == 0 && a.size(0) == 0) {
      c->CopyFrom(b);
      return true;
    }
    CAFFE_ENFORCE(&a == c, "First argument must be in-place.");
    CAFFE_ENFORCE(c->dim() == b.dim());
    CAFFE_ENFORCE(b.dim() == c->dim());
    CAFFE_ENFORCE(a.dtype() == b.dtype());
    for (int i = 1; i < a.dim(); ++i) {
      CAFFE_ENFORCE(a.sizes()[i] == b.sizes()[i]);
    }
    auto oldSize = c->numel();
    c->Extend(b.sizes()[0], kDatasetGrowthPct);
    auto* dst = (char*)c->raw_mutable_data() + oldSize * b.dtype().itemsize();
    context_.CopyItemsSameDevice(b.dtype(), b.numel(), b.raw_data(), dst);
    return true;
  }
};

template <class Context>
class AtomicAppendOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AtomicAppendOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& mutex = OperatorBase::Input<std::unique_ptr<std::mutex>>(0);
    const auto numFields = (InputSize() - 1) / 2;
    CAFFE_ENFORCE(OutputSize() == numFields);

    std::lock_guard<std::mutex> guard(*mutex);

    // 1: checks
    for (int i = 0; i < numFields; ++i) {
      auto& a = Input(1 + i);
      auto& b = Input(1 + i + numFields);
      auto* c = Output(i);
      CAFFE_ENFORCE(b.dim() >= 1);
      if (a.numel() == 0) {
        continue;
      }
      CAFFE_ENFORCE(
          (void*)&a == (void*)c, "Appended-to arguments must be in-place.");
      CAFFE_ENFORCE(c->dim() == b.dim());
      CAFFE_ENFORCE(b.dim() == c->dim());
      CAFFE_ENFORCE(a.dtype() == b.dtype());
      for (int j = 1; j < a.dim(); ++j) {
        CAFFE_ENFORCE(a.sizes()[j] == b.sizes()[j]);
      }
    }

    // 2: copies
    for (int i = 0; i < numFields; ++i) {
      auto& a = Input(1 + i);
      auto& b = Input(1 + i + numFields);
      auto* c = Output(i);
      if (a.numel() == 0 && a.size(0) == 0) {
        c->CopyFrom(b);
        continue;
      }
      auto oldSize = c->numel();
      c->Extend(b.sizes()[0], kDatasetGrowthPct);
      auto* dst = (char*)c->raw_mutable_data() + oldSize * b.dtype().itemsize();
      context_.CopyItemsSameDevice(b.dtype(), b.numel(), b.raw_data(), dst);
    }
    return true;
  }
};

template <class Context>
class CreateTensorVectorOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;

  bool RunOnDevice() override {
    auto ptr = make_unique<std::vector<Tensor>>();
    *OperatorBase::Output<TensorVectorPtr>(TENSOR_VECTOR) = std::move(ptr);
    return true;
  }

 private:
  OUTPUT_TAGS(TENSOR_VECTOR);
};

template <class Context>
class TensorVectorSizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(TensorVectorSizeOp);

  bool RunOnDevice() override {
    auto& vector_ptr = OperatorBase::Input<TensorVectorPtr>(TENSOR_VECTOR);
    auto* size = Output(SIZE);
    size->Resize();
    // 32-bit should be enough here
    *size->template mutable_data<int32_t>() = vector_ptr->size();
    return true;
  }

 private:
  INPUT_TAGS(TENSOR_VECTOR);
  OUTPUT_TAGS(SIZE);
};

template <class Context>
class ConcatTensorVectorOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;

  bool RunOnDevice() override {
    const TensorVectorPtr& tensorVector =
        OperatorBase::Input<TensorVectorPtr>(TENSOR_VECTOR);

    auto* tensor = Output(TENSOR);
    CAFFE_ENFORCE(!tensorVector->empty());

    vector<int64_t> outputDims(tensorVector->at(0).sizes().vec());
    CAFFE_ENFORCE(outputDims.size() > 0);
    for (int i = 1; i < tensorVector->size(); i++) {
      // the tensor shapes are the same except for the first dimension
      for (int j = 1; j < tensorVector->at(i).dim(); j++) {
        CAFFE_ENFORCE(outputDims[j] == tensorVector->at(i).sizes()[j]);
      }
      CAFFE_ENFORCE(tensorVector->at(0).dtype() == tensorVector->at(i).dtype());
      outputDims[0] += tensorVector->at(i).sizes()[0];
    }

    tensor->Resize(outputDims);
    int64_t offset = 0;
    auto* dst = (char*)tensor->raw_mutable_data(tensorVector->at(0).dtype());

    for (const auto& t : *tensorVector) {
      context_.CopyItemsSameDevice(
          t.dtype(), t.numel(), t.raw_data(), dst + offset);
      offset += t.nbytes();
    }

    return true;
  }

 private:
  INPUT_TAGS(TENSOR_VECTOR);
  OUTPUT_TAGS(TENSOR);
};

template <class Context>
class CollectTensorOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CollectTensorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        numToCollect_(
            OperatorBase::GetSingleArgument<int>("num_to_collect", -1)),
        numVisited_(0) {
    CAFFE_ENFORCE(numToCollect_ > 0);
  }

  bool RunOnDevice() override {
    int pos = -1;
    if (numVisited_ < numToCollect_) {
      // append
      pos = numVisited_;
    } else {
      auto& gen = context_.RandGenerator();
      // uniform between [0, numVisited_]
      std::uniform_int_distribution<int> uniformDist(0, numVisited_);
      pos = uniformDist(gen);
      if (pos >= numToCollect_) {
        // discard
        pos = -1;
      }
    }

    for (int i = 0; i < OutputSize(); ++i) {
      // TENSOR_VECTOR_IN is enforced inplace with TENSOR_VECTOR_OUT
      TensorVectorPtr& tensorVector = *OperatorBase::Output<TensorVectorPtr>(i);

      if (numVisited_ >= numToCollect_) {
        CAFFE_ENFORCE(
            tensorVector->size() == numToCollect_,
            "TensorVecotor size = ",
            tensorVector->size(),
            " is different from numToCollect = ",
            numToCollect_);
      }

      const auto& tensor = Input(OutputSize() + i);

      if (pos < 0) {
        // discard
        CAFFE_ENFORCE(numVisited_ >= numToCollect_);
      } else if (pos >= tensorVector->size()) {
        // append
        tensorVector->emplace_back();
        ReinitializeAndCopyFrom(
            &tensorVector->back(),
            Context::GetDeviceType(),
            tensor); // sync copy
      } else {
        // replace
        tensorVector->at(pos).CopyFrom(tensor); // sync copy
      }
    }

    numVisited_++;
    return true;
  }

 private:
  // number of tensors to collect
  int numToCollect_;
  // number of tensors visited
  int numVisited_;
};

class TrimDatasetOp : public Operator<CPUContext> {
 public:
  TrimDatasetOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws),
        iterator_(OperatorBase::GetRepeatedArgument<std::string>("fields")),
        multiple_of_(OperatorBase::GetSingleArgument<int>("multiple_of", 1)) {
    CAFFE_ENFORCE_GE(multiple_of_, 1);
  }

  bool RunOnDevice() override {
    TreeCursor cursor(iterator_);
    TreeWalker walker(Inputs(), cursor);

    int trimmedSize = (walker.size() / multiple_of_) * multiple_of_;
    if (trimmedSize == walker.size()) {
      // we already satisfy the condition
      return true;
    }
    // advance desired number of records
    for (int i = 0; i < trimmedSize; ++i) {
      walker.advance();
    }
    // trim each column to the offset
    for (int col = 0; col < walker.fields().size(); ++col) {
      auto newOuterSize = walker.fields().at(col).offset();
      Output(col)->ShrinkTo(newOuterSize);
    }
    return true;
  }

 private:
  TreeIterator iterator_;
  int multiple_of_;
};

REGISTER_CPU_OPERATOR(CreateTreeCursor, CreateTreeCursorOp);
REGISTER_CPU_OPERATOR(ResetCursor, ResetCursorOp);
REGISTER_CPU_OPERATOR(ReadNextBatch, ReadNextBatchOp);
REGISTER_CPU_OPERATOR(GetCursorOffset, GetCursorOffsetOp);
REGISTER_CPU_OPERATOR(ComputeOffset, ComputeOffsetOp);
REGISTER_CPU_OPERATOR(SortAndShuffle, SortAndShuffleOp);
REGISTER_CPU_OPERATOR(ReadRandomBatch, ReadRandomBatchOp);
REGISTER_CPU_OPERATOR(CheckDatasetConsistency, CheckDatasetConsistencyOp);
REGISTER_CPU_OPERATOR(Append, AppendOp<CPUContext>);
REGISTER_CPU_OPERATOR(AtomicAppend, AtomicAppendOp<CPUContext>);
REGISTER_CPU_OPERATOR(CreateTensorVector, CreateTensorVectorOp<CPUContext>);
REGISTER_CPU_OPERATOR(TensorVectorSize, TensorVectorSizeOp<CPUContext>);
REGISTER_CPU_OPERATOR(ConcatTensorVector, ConcatTensorVectorOp<CPUContext>);
REGISTER_CPU_OPERATOR(CollectTensor, CollectTensorOp<CPUContext>);
REGISTER_CPU_OPERATOR(PackRecords, PackRecordsOp);
REGISTER_CPU_OPERATOR(UnPackRecords, UnPackRecordsOp);
REGISTER_CPU_OPERATOR(TrimDataset, TrimDatasetOp);

OPERATOR_SCHEMA(CreateTreeCursor)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Creates a cursor to iterate through a list of tensors, where some of those
tensors contains the lengths in a nested schema. The schema is determined by
the `fields` arguments.

For example, to represent the following schema:

  Struct(
      a=Int(),
      b=List(List(Int),
      c=List(
          Struct(
             c1=String,
             c2=List(Int),
          ),
      ),
  )

the field list will be:
  [
      "a",
      "b:lengths",
      "b:values:lengths",
      "b:values:values",
      "c:lengths",
      "c:c1",
      "c:c2:lengths",
      "c:c2:values",
  ]

And for the following instance of the struct:

  Struct(
      a=3,
      b=[[4, 5], [6, 7, 8], [], [9]],
      c=[
          Struct(c1='alex', c2=[10, 11]),
          Struct(c1='bob', c2=[12]),
      ],
  )

The values of the fields will be:
  {
      "a": [3],
      "b:lengths": [4],
      "b:values:lengths": [2, 3, 0, 1],
      "b:values:values": [4, 5, 6, 7, 8, 9],
      "c:lengths": [2],
      "c:c1": ["alex", "bob"],
      "c:c2:lengths": [2, 1],
      "c:c2:values", [10, 11, 12],
  }

In general, every field name in the format "{prefix}:lengths" defines a domain
"{prefix}", and every subsequent field in the format "{prefix}:{field}" will
be in that domain, and the length of the domain is provided for each entry of
the parent domain. In the example, "b:lengths" defines a domain of length 4, so
every field under domain "b" will have 4 entries.
The "lengths" field for a given domain must appear before any reference to
that domain.

Returns a pointer to an instance of the Cursor, which keeps the current offset
on each of the domains defined by `fields`. Cursor also ensures thread-safety
such that ReadNextBatch and ResetCursor can be used safely in parallel.

A cursor does not contain data per se, so calls to ReadNextBatch actually need
to pass a list of blobs containing the data to read for each one of the fields.
)DOC")
    .Output(0, "cursor", "A blob pointing to an instance of a new TreeCursor.")
    .Arg(
        "fields",
        "A list of strings each one representing a field of the dataset.");

OPERATOR_SCHEMA(ResetCursor)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Resets the offsets for the given TreeCursor. This operation is thread safe.
)DOC")
    .Input(0, "cursor", "A blob containing a pointer to the cursor.");

OPERATOR_SCHEMA(ReadNextBatch)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Read the next batch of examples out of the given cursor and data blobs.

Input(0) is a blob pointing to a TreeCursor, and
[Input(1),... Input(num_fields)] a list of tensors containing the data for
each field of the dataset.

ReadNextBatch is thread safe.
)DOC")
    .Input(0, "cursor", "A blob containing a pointer to the cursor.")
    .Input(1, "dataset_field_0", "First dataset field")
    .Output(0, "field_0", "Tensor containing the next batch for field 0.")
    .Arg("batch_size", "Number of top-level entries to read.");

OPERATOR_SCHEMA(GetCursorOffset)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Get the current offset in the cursor.")
    .Input(0, "cursor", "A blob containing a pointer to the cursor.")
    .Output(0, "offsets", "Tensor containing the offsets for the cursor.");

OPERATOR_SCHEMA(ComputeOffset)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Compute the offsets matrix given cursor and data blobs. Need to be ran at
beginning or after reseting cursor

Input(0) is a blob pointing to a TreeCursor, and
[Input(1),... Input(num_fields)] a list of tensors containing the data for
each field of the dataset.

ComputeOffset is thread safe.
)DOC")
    .Input(0, "cursor", "A blob containing a pointer to the cursor.")
    .Input(1, "dataset_field_0", "First dataset field")
    .Output(0, "field_0", "Tensor containing offset info for this chunk.");

OPERATOR_SCHEMA(SortAndShuffle)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Compute the sorted indices given a field index to sort by and break the sorted
indices into chunks of shuffle_size * batch_size and shuffle each chunk,
finally we shuffle between batches. If sort_by_field_idx is -1 we skip sort.

For example, we have data sorted as
1,2,3,4,5,6,7,8,9,10,11,12

and batchSize = 2 and shuffleSize = 3, when we shuffle we get:
[3,1,4,6,5,2] [12,10,11,8,9,7]

After this we will shuffle among different batches with size 2
[3,1],[4,6],[5,2],[12,10],[11,8],[9,7]

We may end up with something like
[9,7],[5,2],[12,10],[4,6],[3,1],[11,8]

Input(0) is a blob pointing to a TreeCursor, and
[Input(1),... Input(num_fields)] a list of tensors containing the data for
each field of the dataset.

SortAndShuffle is thread safe.
)DOC")
    .Input(0, "cursor", "A blob containing a pointer to the cursor.")
    .Input(1, "dataset_field_0", "First dataset field")
    .Output(0, "indices", "Tensor containing sorted indices.");

OPERATOR_SCHEMA(ReadRandomBatch)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Read the next batch of examples out of the given cursor,
idx blob, offset matrix and data blobs.

Input(0) is a blob pointing to a TreeCursor,
Input(1) is a blob pointing to the shuffled idx
Input(2) is a blob pointing to the offset matrix and
[Input(3),... Input(num_fields)] a list of tensors containing the data for
each field of the dataset.

ReadRandomBatch is thread safe.
)DOC")
    .Input(0, "cursor", "A blob containing a pointer to the cursor.")
    .Input(1, "idx", "idx with a shuffled order.")
    .Input(2, "offsetsmat", "offset matrix containing length offset info.")
    .Input(3, "dataset_field_0", "First dataset field")
    .Output(0, "field_0", "Tensor containing the next batch for field 0.")
    .Arg("batch_size", "Number of top-level entries to read.")
    .Arg("loop_over", "(bool) Repeat the dataset indefinitely");

OPERATOR_SCHEMA(CheckDatasetConsistency)
    .NumInputs(1, INT_MAX)
    .NumOutputs(0)
    .SetDoc(R"DOC(
Checks that the given data fields represents a consistent dataset under
the schema specified by the `fields` argument. Operator fails if the fields
are not consistent. If data is consistent, each field's data can be safely
appended to an existing dataset, keeping it consistent.
)DOC")
    .Input(0, "field_0", "Data for field 0.")
    .Arg(
        "fields",
        "List of strings representing the string names in the format"
        "specified in the doc for CreateTreeCursor.");

OPERATOR_SCHEMA(Append)
    .NumInputs(2)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Append input `B` to the end of input `A`.

- It is required that this operation run in-place, meaning that the input `A` blob must match the output blob.
- All except the outer-most dimension must be the same between `A` and `B`.
- Input `A` may have to be re-allocated in order for accommodate to the new size. Currently, an exponential growth ratio is used in order to ensure amortized constant time complexity.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/dataset_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Append",
    ["A", "B"],
    ["A"],
)

workspace.FeedBlob("A", np.random.randint(10, size=(1,3,3)))
workspace.FeedBlob("B", np.random.randint(10, size=(2,3,3)))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("A:", workspace.FetchBlob("A"))

```

**Result**

```

A:
[[[3 8 7]
  [1 6 6]
  [5 0 6]]]
B:
[[[4 3 1]
  [7 9 6]
  [9 4 5]]

 [[7 7 4]
  [9 8 7]
  [1 6 6]]]
A:
[[[3 8 7]
  [1 6 6]
  [5 0 6]]

 [[4 3 1]
  [7 9 6]
  [9 4 5]]

 [[7 7 4]
  [9 8 7]
  [1 6 6]]]

```

</details>

)DOC")
    .Input(0, "A", "(*Tensor*): base input tensor of shape $(N, d_1, d_2, ..., d_n)$")
    .Input(1, "B", "(*Tensor*): second input tensor of shape $(M, d_1, d_2, ..., d_n)$ to be appended to the base")
    .Output(0, "A", "(*Tensor*): output tensor of shape $(N+M, d_1, d_2, ..., d_n)$");

OPERATOR_SCHEMA(AtomicAppend)
    .NumInputs(3, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .AllowInplace([](int in, int out) { return in == out + 1; });

OPERATOR_SCHEMA(CreateTensorVector)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Create a std::unique_ptr<std::vector<Tensor> >");

OPERATOR_SCHEMA(TensorVectorSize)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Get the size of the input vector")
    .Input(0, "tensor vector", "std::unique_ptr<std::vector<Tensor> >")
    .Output(0, "size", "int32_t size");

OPERATOR_SCHEMA(ConcatTensorVector)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Concat Tensors in the std::unique_ptr<std::vector<Tensor> >
along the first dimension.
    )DOC")
    .Input(0, "vector of Tensor", "std::unique_ptr<std::vector<Tensor> >")
    .Output(0, "tensor", "tensor after concatenating");

OPERATOR_SCHEMA(CollectTensor)
    .NumInputs([](int n) { return n > 0 && n % 2 == 0; })
    .NumOutputs(1, INT_MAX)
    .NumInputsOutputs([](int in, int out) { return in == out * 2; })
    .EnforceInplace([](int in, int out) { return in == out; })
    .SetDoc(R"DOC(
Collect tensor into tensor vector by reservoir sampling,
argument num_to_collect indicates the max number of tensors that will be
collected. The first half of the inputs are tensor vectors, which are also the
outputs. The second half of the inputs are the tensors to be collected into each
vector (in the same order). The input tensors are collected in all-or-none
manner. If they are collected, they will be placed at the same index in the
output vectors.
)DOC")
    .Arg("num_to_collect", "The max number of tensors to collect");

OPERATOR_SCHEMA(PackRecords)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a dataset under a schema specified by the `fields` argument will pack all
the input tensors into one, where each tensor element represents a row of data
(batch of size 1). This format allows easier use with the rest of Caffe2
operators.
)DOC")
    .Arg(
        "fields",
        "List of strings representing the string names in the format"
        "specified in the doc for CreateTreeCursor.")
    .Output(
        0,
        "tensor",
        "One dimensional tensor having a complex type of SharedTensorVectorPtr."
        " In order to reverse it back to the original input it has to be "
        "inserted into UnPackRecordsOp.");

OPERATOR_SCHEMA(TrimDataset)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Trim the given dataset inplace, given the dataset blobs and the field specs.
Trimming happens such that the dataset will contain the largest possible number
of records that is a multiple of the 'multiple_of' argument.
)DOC")
    .EnforceInplace([](int input, int output) { return input == output; })
    .Arg(
        "fields",
        "List of strings representing the string names in the format"
        "specified in the doc for CreateTreeCursor.");

OPERATOR_SCHEMA(UnPackRecords)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Given a packed dataset (packed by the PackRecordsOp) and the `fields` argument
describing the datasets schema returns the original dataset format. Number of
returned tensors is equal to the number of fields in the `fields` argument.

The first input is the packed tensor to be unpacked. Optionally, you can provide
prototype tensors to give the expected shapes of the output tensors. This is
helpful when you expected to unpack empty tensor, e.g., output of a sampling
process.
)DOC")
    .Arg(
        "fields",
        "List of strings representing the string names in the format"
        "specified in the doc for CreateTreeCursor.")
    .Input(0, "packed_tensor", "The tensor to be unpacked");

SHOULD_NOT_DO_GRADIENT(CreateTreeCursor);
SHOULD_NOT_DO_GRADIENT(ResetCursor);
SHOULD_NOT_DO_GRADIENT(ReadNextBatch);
SHOULD_NOT_DO_GRADIENT(ComputeOffset);
SHOULD_NOT_DO_GRADIENT(ReadRandomBatch);
SHOULD_NOT_DO_GRADIENT(CheckDatasetConsistency);
SHOULD_NOT_DO_GRADIENT(Append);
SHOULD_NOT_DO_GRADIENT(AtomicAppend);
SHOULD_NOT_DO_GRADIENT(CreateTensorVector);
SHOULD_NOT_DO_GRADIENT(TensorVectorSize);
SHOULD_NOT_DO_GRADIENT(ConcatTensorVector);
SHOULD_NOT_DO_GRADIENT(CollectTensor);
SHOULD_NOT_DO_GRADIENT(UnPackRecords);
SHOULD_NOT_DO_GRADIENT(PackRecords);

class TreeCursorSerializer : public BlobSerializerBase {
 public:
  TreeCursorSerializer() {}
  ~TreeCursorSerializer() override {}

  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<std::unique_ptr<TreeCursor>>());
    const auto& cursor =
        *static_cast<const std::unique_ptr<TreeCursor>*>(pointer);
    BlobProto blob_proto;

    // serialize offsets as a tensor
    if (cursor->offsets.size() > 0) {
      Blob offsets_blob;
      auto* offsets = BlobGetMutableTensor(&offsets_blob, CPU);
      offsets->Resize(cursor->offsets.size());
      std::copy(
          cursor->offsets.begin(),
          cursor->offsets.end(),
          offsets->template mutable_data<TOffset>());
      TensorSerializer ser;
      ser.Serialize(
          *offsets, name, blob_proto.mutable_tensor(), 0, offsets->numel());
    }
    blob_proto.set_name(name);
    blob_proto.set_type("std::unique_ptr<TreeCursor>");

    // serialize field names in the content
    std::ostringstream os;
    for (const auto& field : cursor->it.fields()) {
      os << field.name << " ";
    }
    blob_proto.set_content(os.str());

    acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  }
};

class TreeCursorDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    // Deserialize the field names
    std::vector<std::string> fieldNames;
    std::istringstream is(proto.content());
    std::string field;
    while (true) {
      is >> field;
      if (is.eof()) {
        break;
      }
      fieldNames.push_back(field);
    }
    TreeIterator it(fieldNames);

    auto* base = blob->template GetMutable<std::unique_ptr<TreeCursor>>();
    CAFFE_ENFORCE(base != nullptr, "TreeCursor doesn't exist.");
    (*base).reset(new TreeCursor(it));

    // Deserialize the offset vector when it is not empty. The proto.tensor()
    // function will return a TensorProto associated with offset vector. The
    // offset vector contains fields of type int64_t, and we verify it is not
    // empty before calling the deserializer.
    if (proto.tensor().int64_data().size() > 0) {
      TensorDeserializer deser;
      Blob offset_blob;
      deser.Deserialize(proto, &offset_blob);
      auto& offsets = offset_blob.template Get<Tensor>();
      auto* offsets_ptr = offsets.data<TOffset>();
      (*base)->offsets.assign(offsets_ptr, offsets_ptr + offsets.numel());
    }
  }
};

REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<std::unique_ptr<TreeCursor>>()),
    TreeCursorSerializer);
REGISTER_BLOB_DESERIALIZER(std::unique_ptr<TreeCursor>, TreeCursorDeserializer);

} // namespace

void SharedTensorVectorPtrSerializer::Serialize(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  /* This is dummy serialize that doesn't save anything. If saving the content
  is desired in future use case, you can change this serializer. Note: special
  care need to be taken for the parameter initialization of
  LastNWindowCollectorOp and ReservoirSamplingOp if this serializer actually
  saves the content.
  */
  CAFFE_ENFORCE(typeMeta.Match<std::shared_ptr<std::vector<TensorCPU>>>());
  BlobProto blob_proto;
  blob_proto.set_name(name);
  blob_proto.set_type("std::shared_ptr<std::vector<TensorCPU>>");
  blob_proto.set_content("");
  acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
};

void SharedTensorVectorPtrDeserializer::Deserialize(
    const BlobProto& /* unused */,
    Blob* blob) {
  /* This is dummy deserialize which creates a nullptr
   */
  blob->GetMutable<std::shared_ptr<std::vector<TensorCPU>>>();
}

REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<std::shared_ptr<std::vector<TensorCPU>>>()),
    SharedTensorVectorPtrSerializer);

REGISTER_BLOB_DESERIALIZER(
    std::shared_ptr<std::vector<TensorCPU>>,
    SharedTensorVectorPtrDeserializer);

} // namespace dataset_ops
} // namespace caffe2
