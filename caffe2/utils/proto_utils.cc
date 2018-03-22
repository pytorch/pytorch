#include "caffe2/utils/proto_utils.h"

#include <fcntl.h>
#include <cerrno>
#include <fstream>

#include <google/protobuf/io/coded_stream.h>

#ifndef CAFFE2_USE_LITE_PROTO
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#else
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#endif  // !CAFFE2_USE_LITE_PROTO

#include "caffe2/core/logging.h"

using ::google::protobuf::MessageLite;

namespace caffe {

// Caffe wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}

}  // namespace caffe

namespace caffe2 {

// Caffe2 wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}

void ShutdownProtobufLibrary() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

std::string DeviceTypeName(const int32_t& d) {
  switch (d) {
    case CPU:
      return "CPU";
    case CUDA:
      return "CUDA";
    case OPENGL:
      return "OPENGL";
    case MKLDNN:
      return "MKLDNN";
    default:
      CAFFE_THROW(
          "Unknown device: ",
          d,
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the TensorDeviceTypeName() "
          "function to reflect such recent changes?");
      // The below code won't run but is needed to suppress some compiler
      // warnings.
      return "";
  }
};

bool IsSameDevice(const DeviceOption& lhs, const DeviceOption& rhs) {
  return (
      lhs.device_type() == rhs.device_type() &&
      lhs.cuda_gpu_id() == rhs.cuda_gpu_id() &&
      lhs.node_name() == rhs.node_name() &&
      lhs.numa_node_id() == rhs.numa_node_id());
}

bool ReadStringFromFile(const char* filename, string* str) {
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

bool WriteStringToFile(const string& str, const char* filename) {
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
  ~IfstreamInputStream() { ifs_.close(); }

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
}  // namespace

string ProtoDebugString(const MessageLite& proto) {
  return proto.SerializeAsString();
}

bool ParseProtoFromLargeString(const string& str, MessageLite* proto) {
  ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  // Set PlanDef message size limit to 1G.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(filename));
  stream.SetOwnsCopyingStream(true);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

void WriteProtoToBinaryFile(
    const MessageLite& /*proto*/,
    const char* /*filename*/) {
  LOG(FATAL) << "Not implemented yet.";
}

#else  // CAFFE2_USE_LITE_PROTO

// Full protocol buffer.

using ::google::protobuf::io::FileInputStream;
using ::google::protobuf::io::FileOutputStream;
using ::google::protobuf::io::ZeroCopyInputStream;
using ::google::protobuf::io::CodedInputStream;
using ::google::protobuf::io::ZeroCopyOutputStream;
using ::google::protobuf::io::CodedOutputStream;
using ::google::protobuf::Message;

namespace TextFormat {
bool ParseFromString(const string& spec, Message* proto) {
  return ::google::protobuf::TextFormat::ParseFromString(spec, proto);
}
} // namespace TextFormat

string ProtoDebugString(const Message& proto) {
  return proto.ShortDebugString();
}

bool ParseProtoFromLargeString(const string& str, Message* proto) {
  ::google::protobuf::io::ArrayInputStream input_stream(str.data(), str.size());
  ::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
  // Set PlanDef message size limit to 1G.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CAFFE_ENFORCE_NE(fd, -1, "File not found: ", filename);
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CAFFE_ENFORCE(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto) {
#if defined (_MSC_VER)  // for MSC compiler binary flag needs to be specified
  int fd = open(filename, O_RDONLY | O_BINARY);
#else
  int fd = open(filename, O_RDONLY);
#endif
  CAFFE_ENFORCE_NE(fd, -1, "File not found: ", filename);
  std::unique_ptr<ZeroCopyInputStream> raw_input(new FileInputStream(fd));
  std::unique_ptr<CodedInputStream> coded_input(
      new CodedInputStream(raw_input.get()));
  // A hack to manually allow using very large protocol buffers.
  coded_input->SetTotalBytesLimit(1073741824, 536870912);
  bool success = proto->ParseFromCodedStream(coded_input.get());
  coded_input.reset();
  raw_input.reset();
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename) {
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

#endif  // CAFFE2_USE_LITE_PROTO


ArgumentHelper::ArgumentHelper(const OperatorDef& def) {
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
                     << "] found in operator def: "
                     << ProtoDebugString(def);
      }
    }
    arg_map_[arg.name()] = arg;
  }
}

ArgumentHelper::ArgumentHelper(const NetDef& netdef) {
  for (auto& arg : netdef.arg()) {
    CAFFE_ENFORCE(
        arg_map_.count(arg.name()) == 0,
        "Duplicated argument name [", arg.name(), "] found in net def: ",
        ProtoDebugString(netdef));
    arg_map_[arg.name()] = arg;
  }
}

bool ArgumentHelper::HasArgument(const string& name) const {
  return arg_map_.count(name);
}

namespace {
// Helper function to verify that conversion between types won't loose any
// significant bit.
template <typename InputType, typename TargetType>
bool SupportsLosslessConversion(const InputType& value) {
  return static_cast<InputType>(static_cast<TargetType>(value)) == value;
}
}

bool operator==(const NetDef& l, const NetDef& r) {
  return l.SerializeAsString() == r.SerializeAsString();
}

std::ostream& operator<<(std::ostream& output, const NetDef& n) {
  output << n.SerializeAsString();
  return output;
}

#define INSTANTIATE_GET_SINGLE_ARGUMENT(                                      \
    T, fieldname, enforce_lossless_conversion)                                \
  template <>                                                                 \
  T ArgumentHelper::GetSingleArgument<T>(                                     \
      const string& name, const T& default_value) const {                     \
    if (arg_map_.count(name) == 0) {                                          \
      VLOG(1) << "Using default parameter value " << default_value            \
              << " for parameter " << name;                                   \
      return default_value;                                                   \
    }                                                                         \
    CAFFE_ENFORCE(                                                            \
        arg_map_.at(name).has_##fieldname(),                                  \
        "Argument ",                                                          \
        name,                                                                 \
        " does not have the right field: expected field " #fieldname);        \
    auto value = arg_map_.at(name).fieldname();                               \
    if (enforce_lossless_conversion) {                                        \
      auto supportsConversion =                                               \
          SupportsLosslessConversion<decltype(value), T>(value);              \
      CAFFE_ENFORCE(                                                          \
          supportsConversion,                                                 \
          "Value",                                                            \
          value,                                                              \
          " of argument ",                                                    \
          name,                                                               \
          "cannot be represented correctly in a target type");                \
    }                                                                         \
    return static_cast<T>(value);                                             \
  }                                                                           \
  template <>                                                                 \
  bool ArgumentHelper::HasSingleArgumentOfType<T>(const string& name) const { \
    if (arg_map_.count(name) == 0) {                                          \
      return false;                                                           \
    }                                                                         \
    return arg_map_.at(name).has_##fieldname();                               \
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
  vector<T> ArgumentHelper::GetRepeatedArgument<T>(                    \
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
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

#define CAFFE2_MAKE_SINGULAR_ARGUMENT(T, fieldname)                            \
template <>                                                                    \
Argument MakeArgument(const string& name, const T& value) {                    \
  Argument arg;                                                                \
  arg.set_name(name);                                                          \
  arg.set_##fieldname(value);                                                  \
  return arg;                                                                  \
}

CAFFE2_MAKE_SINGULAR_ARGUMENT(bool, i)
CAFFE2_MAKE_SINGULAR_ARGUMENT(float, f)
CAFFE2_MAKE_SINGULAR_ARGUMENT(int, i)
CAFFE2_MAKE_SINGULAR_ARGUMENT(int64_t, i)
CAFFE2_MAKE_SINGULAR_ARGUMENT(string, s)
#undef CAFFE2_MAKE_SINGULAR_ARGUMENT

template <>
Argument MakeArgument(const string& name, const MessageLite& value) {
  Argument arg;
  arg.set_name(name);
  arg.set_s(value.SerializeAsString());
  return arg;
}

#define CAFFE2_MAKE_REPEATED_ARGUMENT(T, fieldname)                            \
template <>                                                                    \
Argument MakeArgument(const string& name, const vector<T>& value) {            \
  Argument arg;                                                                \
  arg.set_name(name);                                                          \
  for (const auto& v : value) {                                                \
    arg.add_##fieldname(v);                                                    \
  }                                                                            \
  return arg;                                                                  \
}

CAFFE2_MAKE_REPEATED_ARGUMENT(float, floats)
CAFFE2_MAKE_REPEATED_ARGUMENT(int, ints)
CAFFE2_MAKE_REPEATED_ARGUMENT(int64_t, ints)
CAFFE2_MAKE_REPEATED_ARGUMENT(string, strings)
#undef CAFFE2_MAKE_REPEATED_ARGUMENT

bool HasOutput(const OperatorDef& op, const std::string& output) {
  for (const auto& outp : op.output()) {
    if (outp == output) {
      return true;
    }
  }
  return false;
}

bool HasInput(const OperatorDef& op, const std::string& input) {
  for (const auto& inp : op.input()) {
    if (inp == input) {
      return true;
    }
  }
  return false;
}

const Argument& GetArgument(const OperatorDef& def, const string& name) {
  for (const Argument& arg : def.arg()) {
    if (arg.name() == name) {
      return arg;
    }
  }
  CAFFE_THROW(
      "Argument named ",
      name,
      " does not exist in operator ",
      ProtoDebugString(def));
}

bool GetFlagArgument(
    const OperatorDef& def,
    const string& name,
    bool def_value) {
  for (const Argument& arg : def.arg()) {
    if (arg.name() == name) {
      CAFFE_ENFORCE(
          arg.has_i(), "Can't parse argument as bool: ", ProtoDebugString(arg));
      return arg.i();
    }
  }
  return def_value;
}

Argument* GetMutableArgument(
    const string& name,
    const bool create_if_missing,
    OperatorDef* def) {
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

}  // namespace caffe2
