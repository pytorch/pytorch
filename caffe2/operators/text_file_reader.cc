#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/text_file_reader_utils.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

struct TextFileReaderInstance {
  TextFileReaderInstance(
      const std::vector<char>& delims,
      char escape,
      const std::string& filename,
      int numPasses,
      const std::vector<int>& types)
      : fileReader(filename),
        tokenizer(Tokenizer(delims, escape), &fileReader, numPasses),
        fieldTypes(types) {
    for (const auto dt : fieldTypes) {
      fieldMetas.push_back(
          DataTypeToTypeMeta(static_cast<TensorProto_DataType>(dt)));
      fieldByteSizes.push_back(fieldMetas.back().itemsize());
    }
  }

  FileReader fileReader;
  BufferedTokenizer tokenizer;
  std::vector<int> fieldTypes;
  std::vector<TypeMeta> fieldMetas;
  std::vector<size_t> fieldByteSizes;
  size_t rowsRead{0};

  // hack to guarantee thread-safeness of the read op
  // TODO(azzolini): support multi-threaded reading.
  std::mutex globalMutex_;
};

class CreateTextFileReaderOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit CreateTextFileReaderOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        filename_(GetSingleArgument<string>("filename", "")),
        numPasses_(GetSingleArgument<int>("num_passes", 1)),
        fieldTypes_(GetRepeatedArgument<int>("field_types")) {
    CAFFE_ENFORCE(fieldTypes_.size() > 0, "field_types arg must be non-empty");
  }

  bool RunOnDevice() override {
    *OperatorBase::Output<std::unique_ptr<TextFileReaderInstance>>(0) =
        std::unique_ptr<TextFileReaderInstance>(new TextFileReaderInstance(
            {'\n', '\t'}, '\0', filename_, numPasses_, fieldTypes_));
    return true;
  }

 private:
  std::string filename_;
  int numPasses_;
  std::vector<int> fieldTypes_;
};

inline void convert(
    TensorProto_DataType dst_type,
    const char* src_start,
    const char* src_end,
    void* dst) {
  switch (dst_type) {
    case TensorProto_DataType_STRING: {
      static_cast<std::string*>(dst)->assign(src_start, src_end);
    } break;
    case TensorProto_DataType_FLOAT: {
      // TODO(azzolini): avoid copy, use faster conversion
      std::string str_copy(src_start, src_end);
      const char* src_copy = str_copy.c_str();
      char* src_copy_end;
      float val = strtof(src_copy, &src_copy_end);
      if (src_copy == src_copy_end) {
        throw std::runtime_error("Invalid float: " + str_copy);
      }
      *static_cast<float*>(dst) = val;
    } break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

class TextFileReaderReadOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit TextFileReaderReadOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...),
        batchSize_(GetSingleArgument<int>("batch_size", 1)) {}

  bool RunOnDevice() override {
    const int numFields = OutputSize();
    CAFFE_ENFORCE(numFields > 0, "Expected at least one output.");

    auto instance =
        OperatorBase::Input<std::unique_ptr<TextFileReaderInstance>>(0).get();

    CAFFE_ENFORCE(
        instance->fieldTypes.size() == numFields,
        "Invalid number of outputs. Expected " +
            to_string(instance->fieldTypes.size()) + " got " +
            to_string(numFields));

    // char* datas[numFields];
    // MSVC does not allow using const int, so we will need to dynamically allocate
    // it.
    std::vector<char*> datas(numFields);
    for (int i = 0; i < numFields; ++i) {
      Output(i)->Resize(batchSize_);
      datas[i] = (char*)Output(i)->raw_mutable_data(instance->fieldMetas[i]);
    }

    int rowsRead = 0;
    {
      // TODO(azzolini): support multi-threaded reading
      std::lock_guard<std::mutex> guard(instance->globalMutex_);

      bool finished = false;
      Token token;
      while (!finished && (rowsRead < batchSize_)) {
        int field;
        for (field = 0; field < numFields; ++field) {
          finished = !instance->tokenizer.next(token);
          if (finished) {
            CAFFE_ENFORCE(
                field == 0, "Invalid number of fields at end of file.");
            break;
          }
          CAFFE_ENFORCE(
              (field == 0 && token.startDelimId == 0) ||
                  (field > 0 && token.startDelimId == 1),
              "Invalid number of columns at row ",
              instance->rowsRead + rowsRead + 1);
          const auto& meta = instance->fieldMetas[field];
          char*& data = datas[field];
          convert(
              (TensorProto_DataType)instance->fieldTypes[field],
              token.start,
              token.end,
              data);
          data += instance->fieldByteSizes[field];
        }
        if (!finished) {
          ++rowsRead;
        }
      }
      instance->rowsRead += rowsRead;
    }

    for (int i = 0; i < numFields; ++i) {
      Output(i)->ShrinkTo(rowsRead);
    }
    return true;
  }

 private:
  int64_t batchSize_;
};

CAFFE_KNOWN_TYPE(std::unique_ptr<TextFileReaderInstance>);

REGISTER_CPU_OPERATOR(CreateTextFileReader, CreateTextFileReaderOp);
REGISTER_CPU_OPERATOR(TextFileReaderRead, TextFileReaderReadOp);

OPERATOR_SCHEMA(CreateTextFileReader)
    .NumInputs(0)
    .NumOutputs(1)
    .ScalarType(TensorProto::UNDEFINED)
    .SetDoc("Create a text file reader. Fields are delimited by <TAB>.")
    .Arg("filename", "Path to the file.")
    .Arg("num_passes", "Number of passes over the file.")
    .Arg(
        "field_types",
        "List with type of each field. Type enum is found at core.DataType.")
    .Output(0, "handler", "Pointer to the created TextFileReaderInstance.");

OPERATOR_SCHEMA(TextFileReaderRead)
    .NumInputs(1)
    .NumOutputs(1, INT_MAX)
    .SetDoc(
        "Read a batch of rows from the given text file reader instance. "
        "Expects the number of fields to be equal to the number of outputs. "
        "Each output is a 1D tensor containing the values for the given field "
        "for each row. When end of file is reached, returns empty tensors.")
    .Input(0, "handler", "Pointer to an existing TextFileReaderInstance.")
    .Arg("batch_size", "Maximum number of rows to read.");

NO_GRADIENT(CreateTextFileReader);
NO_GRADIENT(TextFileReaderRead);

} // namespace caffe2
