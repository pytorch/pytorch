#include "caffe2/utils/proto_utils.h"

#include <c10/core/DeviceType.h>

#include <fcntl.h>
#include <cerrno>
#include <fstream>
#include <unordered_set>

#include <google/protobuf/io/coded_stream.h>

#ifndef CAFFE2_USE_LITE_PROTO
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#else
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#endif // !CAFFE2_USE_LITE_PROTO

#include "caffe2/core/logging.h"

using ::google::protobuf::MessageLite;

namespace caffe2 {

C10_EXPORT std::string DeviceTypeName(const int32_t& d) {
  return at::DeviceTypeName(static_cast<at::DeviceType>(d));
}

C10_EXPORT int DeviceId(const DeviceOption& option) {
  switch (option.device_type()) {
    case PROTO_CPU:
      return option.numa_node_id();
    case PROTO_CUDA:
    case PROTO_HIP:
      return option.device_id();
    case PROTO_MKLDNN:
      return option.numa_node_id();
    default:
      CAFFE_THROW("Unknown device id for device type: ", option.device_type());
  }
}

C10_EXPORT bool IsSameDevice(const DeviceOption& lhs, const DeviceOption& rhs) {
  return (
      lhs.device_type() == rhs.device_type() &&
      lhs.device_id() == rhs.device_id() &&
      lhs.node_name() == rhs.node_name() &&
      lhs.numa_node_id() == rhs.numa_node_id());
}

C10_EXPORT bool IsCPUDeviceType(int device_type) {
  static const std::unordered_set<int> cpu_types{
      PROTO_CPU,
      PROTO_MKLDNN,
      PROTO_IDEEP,
  };
  return cpu_types.count(device_type);
}

C10_EXPORT bool IsGPUDeviceType(int device_type) {
  static const std::unordered_set<int> gpu_types{
      PROTO_CUDA,
      PROTO_HIP,
  };
  return gpu_types.count(device_type);
}

C10_EXPORT bool ReadStringFromFile(const char* filename, string* str) {
  std::ifstream ifs(filename, std::ios::in);
  if (!ifs) {
    VLOG(1) << "File cannot be opened: " << filename
            << " error: " << ifs.rdstate();
    return false;
  }
  ifs.seekg(0, std::ios::end);
  size_t n = ifs.tellg();
  str->resize(n);
  ifs.seekg(0);
  ifs.read(&(*str)[0], n);
  return true;
}

C10_EXPORT bool WriteStringToFile(const string& str, const char* filename) {
  std::ofstream ofs(filename, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    VLOG(1) << "File cannot be created: " << filename
            << " error: " << ofs.rdstate();
    return false;
  }
  ofs << str;
  return true;
}

// IO-specific proto functions: we will deal with the protocol buffer lite and
// full versions differently.

#ifdef CAFFE2_USE_LITE_PROTO

// Lite runtime.

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const string& filename)
      : ifs_(filename.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() {
    ifs_.close();
  }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
} // namespace

C10_EXPORT string ProtoDebugString(const MessageLite& proto) {
  string serialized = proto.SerializeAsString();
  for (char& c : serialized) {
    if (c < 0x20 || c >= 0x7f) {
      c = '?';
    }
  }
  return serialized;
}

C10_EXPORT bool ParseProtoFromLargeString(
    const string& str,
    MessageLite* proto) {
  ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  // Set PlanDef message size limit to 2G.
  coded_stream.SetTotalBytesLimit(2147483647, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

C10_EXPORT bool ReadProtoFromBinaryFile(
    const char* filename,
    MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(filename));
  stream.SetOwnsCopyingStream(true);
  // Total bytes hard limit / warning limit are set to 2GB and 512MB
  // respectively.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  coded_stream.SetTotalBytesLimit(2147483647, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

C10_EXPORT void WriteProtoToBinaryFile(
    const MessageLite& /*proto*/,
    const char* /*filename*/) {
  LOG(FATAL) << "Not implemented yet.";
}

#else // CAFFE2_USE_LITE_PROTO

// Full protocol buffer.

using ::google::protobuf::Message;
using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::CodedOutputStream;
using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::FileOutputStream;
using ::google::protobuf::io::ZeroCopyInputStream;
using ::google::protobuf::io::ZeroCopyOutputStream;

namespace TextFormat {
C10_EXPORT bool ParseFromString(const string& spec, Message* proto) {
  string bc_spec = spec;

  {
    auto num_replaced = c10::ReplaceAll(bc_spec, "cuda_gpu_id", "device_id");
    if (num_replaced) {
      LOG(ERROR) << "Your model was serialized in Protobuf TextFormat and "
                 << "it has " << num_replaced
                 << " places using the deprecated field name 'cuda_gpu_id'!\n"
                 << spec
                 << "\nPlease re-export your model in Protobuf binary format "
                 << "to make it backward compatible for field renaming.";
    }
  }

  return ::google::protobuf::TextFormat::ParseFromString(
      std::move(bc_spec), proto);
}
} // namespace TextFormat

C10_EXPORT string ProtoDebugString(const Message& proto) {
  return proto.ShortDebugString();
}

C10_EXPORT bool ParseProtoFromLargeString(const string& str, Message* proto) {
  ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  // Set PlanDef message size limit to 2G.
  coded_stream.SetTotalBytesLimit(2147483647, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

C10_EXPORT bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CAFFE_ENFORCE_NE(fd, -1, "File not found: ", filename);
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

C10_EXPORT void WriteProtoToTextFile(
    const Message& proto,
    const char* filename,
    bool throwIfError) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  if(!google::protobuf::TextFormat::Print(proto, output)) {
     if (throwIfError) {
       CAFFE_THROW("Cannot write proto to text file: ", filename);
     } else {
       LOG(ERROR) << "Cannot write proto to text file: " << filename;
     }
  }
  delete output;
  close(fd);
}

C10_EXPORT bool ReadProtoFromBinaryFile(
    const char* filename,
    MessageLite* proto) {
#if defined(_MSC_VER) // for MSC compiler binary flag needs to be specified
  int fd = open(filename, O_RDONLY | O_BINARY);
#else
  int fd = open(filename, O_RDONLY);
#endif
  CAFFE_ENFORCE_NE(fd, -1, "File not found: ", filename);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(
      new CodedInputStream(raw_input.get()));
  // A hack to manually allow using very large protocol buffers.
  coded_input->SetTotalBytesLimit(2147483647, 536870912);
  bool success = proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  close(fd);
  return success;
}

C10_EXPORT void WriteProtoToBinaryFile(
    const MessageLite& proto,
    const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CAFFE_ENFORCE_NE(
      fd, -1, "File cannot be created: ", filename, " error number: ", errno);
  std::unique_ptr<ZeroCopyOutputStream> raw_output(new FileOutputStream(fd));
  std::unique_ptr<CodedOutputStream> coded_output(
      new CodedOutputStream(raw_output.get()));
  CAFFE_ENFORCE(proto.SerializeToCodedStream(coded_output.get()));
  coded_output.reset();
  raw_output.reset();
  close(fd);
}

#endif // CAFFE2_USE_LITE_PROTO

C10_EXPORT ArgumentHelper::ArgumentHelper(const OperatorDef& def) {
  for (auto& arg : def.arg()) {
    if (arg_map_.count(arg.name())) {
      if (arg.SerializeAsString() != arg_map_[arg.name()].SerializeAsString()) {
        // If there are two arguments of the same name but different contents,
        // we will throw an error.
        CAFFE_THROW(
            "Found argument of the same name ",
            arg.name(),
            "but with different contents.",
            ProtoDebugString(def));
      } else {
        LOG(WARNING) << "Duplicated argument name [" << arg.name()
                     << "] found in operator def: " << ProtoDebugString(def);
      }
    }
    arg_map_[arg.name()] = arg;
  }
}

C10_EXPORT ArgumentHelper::ArgumentHelper(const NetDef& netdef) {
  for (auto& arg : netdef.arg()) {
    CAFFE_ENFORCE(
        arg_map_.count(arg.name()) == 0,
        "Duplicated argument name [",
        arg.name(),
        "] found in net def: ",
        ProtoDebugString(netdef));
    arg_map_[arg.name()] = arg;
  }
}

C10_EXPORT bool ArgumentHelper::HasArgument(const string& name) const {
  return arg_map_.count(name);
}

namespace {
// Helper function to verify that conversion between types won't loose any
// significant bit.
template <typename InputType, typename TargetType>
bool SupportsLosslessConversion(const InputType& value) {
  return static_cast<InputType>(static_cast<TargetType>(value)) == value;
}
} // namespace
bool operator==(const TensorProto& l, const TensorProto& r) {
  return l.SerializeAsString() == r.SerializeAsString();
}

std::ostream& operator<<(std::ostream& output, const TensorProto& n) {
  output << n.SerializeAsString();
  return output;
}
bool operator==(const QTensorProto& l, const QTensorProto& r) {
  return l.SerializeAsString() == r.SerializeAsString();
}

std::ostream& operator<<(std::ostream& output, const QTensorProto& n) {
  output << n.SerializeAsString();
  return output;
}
bool operator==(const NetDef& l, const NetDef& r) {
  return l.SerializeAsString() == r.SerializeAsString();
}

std::ostream& operator<<(std::ostream& output, const NetDef& n) {
  output << n.SerializeAsString();
  return output;
}

#define INSTANTIATE_GET_SINGLE_ARGUMENT(                               \
    T, fieldname, enforce_lossless_conversion)                         \
  template <>                                                          \
  C10_EXPORT T ArgumentHelper::GetSingleArgument<T>(                   \
      const string& name, const T& default_value) const {              \
    if (arg_map_.count(name) == 0) {                                   \
      VLOG(1) << "Using default parameter value " << default_value     \
              << " for parameter " << name;                            \
      return default_value;                                            \
    }                                                                  \
    CAFFE_ENFORCE(                                                     \
        arg_map_.at(name).has_##fieldname(),                           \
        "Argument ",                                                   \
        name,                                                          \
        " does not have the right field: expected field " #fieldname); \
    auto value = arg_map_.at(name).fieldname();                        \
    if (enforce_lossless_conversion) {                                 \
      auto supportsConversion =                                        \
          SupportsLosslessConversion<decltype(value), T>(value);       \
      CAFFE_ENFORCE(                                                   \
          supportsConversion,                                          \
          "Value",                                                     \
          value,                                                       \
          " of argument ",                                             \
          name,                                                        \
          "cannot be represented correctly in a target type");         \
    }                                                                  \
    return static_cast<T>(value);                                      \
  }                                                                    \
  template <>                                                          \
  C10_EXPORT bool ArgumentHelper::HasSingleArgumentOfType<T>(          \
      const string& name) const {                                      \
    if (arg_map_.count(name) == 0) {                                   \
      return false;                                                    \
    }                                                                  \
    return arg_map_.at(name).has_##fieldname();                        \
  }

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(double, f, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, i, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(int8_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int16_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(int64_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(uint8_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(uint16_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(size_t, i, true)
INSTANTIATE_GET_SINGLE_ARGUMENT(string, s, false)
INSTANTIATE_GET_SINGLE_ARGUMENT(NetDef, n, false)
#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(                             \
    T, fieldname, enforce_lossless_conversion)                         \
  template <>                                                          \
  C10_EXPORT vector<T> ArgumentHelper::GetRepeatedArgument<T>(         \
      const string& name, const std::vector<T>& default_value) const { \
    if (arg_map_.count(name) == 0) {                                   \
      return default_value;                                            \
    }                                                                  \
    vector<T> values;                                                  \
    for (const auto& v : arg_map_.at(name).fieldname()) {              \
      if (enforce_lossless_conversion) {                               \
        auto supportsConversion =                                      \
            SupportsLosslessConversion<decltype(v), T>(v);             \
        CAFFE_ENFORCE(                                                 \
            supportsConversion,                                        \
            "Value",                                                   \
            v,                                                         \
            " of argument ",                                           \
            name,                                                      \
            "cannot be represented correctly in a target type");       \
      }                                                                \
      values.push_back(static_cast<T>(v));                             \
    }                                                                  \
    return values;                                                     \
  }

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(double, floats, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, ints, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(int8_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int16_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(int64_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(uint8_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(uint16_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(size_t, ints, true)
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(NetDef, nets, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(TensorProto, tensors, false)
INSTANTIATE_GET_REPEATED_ARGUMENT(QTensorProto, qtensors, false)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

#define CAFFE2_MAKE_SINGULAR_ARGUMENT(T, fieldname)                      \
  template <>                                                            \
  C10_EXPORT Argument MakeArgument(const string& name, const T& value) { \
    Argument arg;                                                        \
    arg.set_name(name);                                                  \
    arg.set_##fieldname(value);                                          \
    return arg;                                                          \
  }

CAFFE2_MAKE_SINGULAR_ARGUMENT(bool, i)
CAFFE2_MAKE_SINGULAR_ARGUMENT(float, f)
CAFFE2_MAKE_SINGULAR_ARGUMENT(int, i)
CAFFE2_MAKE_SINGULAR_ARGUMENT(int16_t, i)
CAFFE2_MAKE_SINGULAR_ARGUMENT(int64_t, i)
CAFFE2_MAKE_SINGULAR_ARGUMENT(string, s)
#undef CAFFE2_MAKE_SINGULAR_ARGUMENT

template <>
C10_EXPORT Argument MakeArgument(const string& name, const NetDef& value) {
  Argument arg;
  arg.set_name(name);
  *arg.mutable_n() = value;
  return arg;
}

template <>
C10_EXPORT bool ArgumentHelper::RemoveArgument(OperatorDef& def, int index);
template <>
bool ArgumentHelper::RemoveArgument(NetDef& def, int index);

template <>
C10_EXPORT Argument MakeArgument(const string& name, const MessageLite& value) {
  Argument arg;
  arg.set_name(name);
  arg.set_s(value.SerializeAsString());
  return arg;
}

#define CAFFE2_MAKE_REPEATED_ARGUMENT(T, fieldname) \
  template <>                                       \
  C10_EXPORT Argument MakeArgument(                 \
      const string& name, const vector<T>& value) { \
    Argument arg;                                   \
    arg.set_name(name);                             \
    for (const auto& v : value) {                   \
      arg.add_##fieldname(v);                       \
    }                                               \
    return arg;                                     \
  }

CAFFE2_MAKE_REPEATED_ARGUMENT(float, floats)
CAFFE2_MAKE_REPEATED_ARGUMENT(int, ints)
CAFFE2_MAKE_REPEATED_ARGUMENT(int64_t, ints)
CAFFE2_MAKE_REPEATED_ARGUMENT(string, strings)
#undef CAFFE2_MAKE_REPEATED_ARGUMENT

C10_EXPORT bool HasOutput(const OperatorDef& op, const std::string& output) {
  for (const auto& outp : op.output()) {
    if (outp == output) {
      return true;
    }
  }
  return false;
}

C10_EXPORT bool HasInput(const OperatorDef& op, const std::string& input) {
  for (const auto& inp : op.input()) {
    if (inp == input) {
      return true;
    }
  }
  return false;
}

// Return the argument index or -1 if it does not exist.
C10_EXPORT int GetArgumentIndex(
    const google::protobuf::RepeatedPtrField<Argument>& args,
    const string& name) {
  int index = 0;
  for (const Argument& arg : args) {
    if (arg.name() == name) {
      return index;
    }
    index++;
  }
  return -1;
}

C10_EXPORT const Argument& GetArgument(
    const OperatorDef& def,
    const string& name) {
  int index = GetArgumentIndex(def.arg(), name);
  if (index != -1) {
    return def.arg(index);
  } else {
    CAFFE_THROW(
        "Argument named ",
        name,
        " does not exist in operator ",
        ProtoDebugString(def));
  }
}

C10_EXPORT const Argument& GetArgument(const NetDef& def, const string& name) {
  int index = GetArgumentIndex(def.arg(), name);
  if (index != -1) {
    return def.arg(index);
  } else {
    CAFFE_THROW(
        "Argument named ",
        name,
        " does not exist in net ",
        ProtoDebugString(def));
  }
}

C10_EXPORT const Argument* GetArgumentPtr(
    const OperatorDef& def,
    const string& name) {
  int index = GetArgumentIndex(def.arg(), name);
  if (index != -1) {
    return &def.arg(index);
  } else {
    return nullptr;
  }
}

C10_EXPORT const Argument* GetArgumentPtr(
    const NetDef& def,
    const string& name) {
  int index = GetArgumentIndex(def.arg(), name);
  if (index != -1) {
    return &def.arg(index);
  } else {
    return nullptr;
  }
}

C10_EXPORT bool GetFlagArgument(
    const google::protobuf::RepeatedPtrField<Argument>& args,
    const string& name,
    bool default_value) {
  int index = GetArgumentIndex(args, name);
  if (index != -1) {
    auto arg = args.Get(index);
    CAFFE_ENFORCE(
        arg.has_i(), "Can't parse argument as bool: ", ProtoDebugString(arg));
    return arg.i();
  }
  return default_value;
}

C10_EXPORT bool GetFlagArgument(
    const OperatorDef& def,
    const string& name,
    bool default_value) {
  return GetFlagArgument(def.arg(), name, default_value);
}

C10_EXPORT bool
GetFlagArgument(const NetDef& def, const string& name, bool default_value) {
  return GetFlagArgument(def.arg(), name, default_value);
}

template <typename Def>
Argument* GetMutableArgumentImpl(
    const string& name,
    const bool create_if_missing,
    Def* def) {
  for (int i = 0; i < def->arg_size(); ++i) {
    if (def->arg(i).name() == name) {
      return def->mutable_arg(i);
    }
  }
  // If no argument of the right name is found...
  if (create_if_missing) {
    Argument* arg = def->add_arg();
    arg->set_name(name);
    return arg;
  } else {
    return nullptr;
  }
}

C10_EXPORT Argument* GetMutableArgument(
    const string& name,
    const bool create_if_missing,
    OperatorDef* def) {
  return GetMutableArgumentImpl(name, create_if_missing, def);
}

C10_EXPORT Argument* GetMutableArgument(
    const string& name,
    const bool create_if_missing,
    NetDef* def) {
  return GetMutableArgumentImpl(name, create_if_missing, def);
}

C10_EXPORT void cleanupExternalInputsAndOutputs(NetDef* net) {
  std::vector<std::string> oldExternalInputs;
  for (const auto& input : net->external_input()) {
    oldExternalInputs.emplace_back(input);
  }
  std::vector<std::string> oldExternalOutputs;
  for (const auto& output : net->external_output()) {
    oldExternalOutputs.emplace_back(output);
  }

  net->clear_external_input();
  net->clear_external_output();

  std::set<std::string> inputSet;
  for (const auto& input : oldExternalInputs) {
    if (inputSet.count(input)) {
      // Prevent duplicate external inputs.
      continue;
    }
    inputSet.insert(input);
    net->add_external_input(input);
  }

  // Set of blobs that are external inputs or outputs of some operators.
  std::set<std::string> allOutputs(inputSet.begin(), inputSet.end());
  for (const auto& op : net->op()) {
    for (const auto& input : op.input()) {
      if (inputSet.count(input) || allOutputs.count(input)) {
        continue;
      }
      // Add missing external inputs.
      inputSet.insert(input);
      net->add_external_input(input);
    }
    for (const auto& output : op.output()) {
      allOutputs.insert(output);
    }
  }

  std::set<std::string> outputSet;
  for (const auto& output : oldExternalOutputs) {
    if (!allOutputs.count(output)) {
      continue;
    }
    if (outputSet.count(output)) {
      continue;
    }
    outputSet.insert(output);
    net->add_external_output(output);
  }
}

} // namespace caffe2
