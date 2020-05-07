#ifndef CAFFE2_UTILS_PROTO_UTILS_H_
#define CAFFE2_UTILS_PROTO_UTILS_H_

#ifdef CAFFE2_USE_LITE_PROTO
#include <google/protobuf/message_lite.h>
#else // CAFFE2_USE_LITE_PROTO
#include <google/protobuf/message.h>
#endif  // !CAFFE2_USE_LITE_PROTO

#include "caffe2/core/logging.h"
#include "caffe2/utils/proto_wrap.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

using std::string;
using ::google::protobuf::MessageLite;

// A wrapper function to return device name string for use in blob serialization
// / deserialization. This should have one to one correspondence with
// caffe2/proto/caffe2.proto: enum DeviceType.
//
// Note that we can't use DeviceType_Name, because that is only available in
// protobuf-full, and some platforms (like mobile) may want to use
// protobuf-lite instead.
CAFFE2_API std::string DeviceTypeName(const int32_t& d);

CAFFE2_API int DeviceId(const DeviceOption& option);

// Returns if the two DeviceOptions are pointing to the same device.
CAFFE2_API bool IsSameDevice(const DeviceOption& lhs, const DeviceOption& rhs);

CAFFE2_API bool IsCPUDeviceType(int device_type);
CAFFE2_API bool IsGPUDeviceType(int device_type);

// Common interfaces that reads file contents into a string.
CAFFE2_API bool ReadStringFromFile(const char* filename, string* str);
CAFFE2_API bool WriteStringToFile(const string& str, const char* filename);

// Common interfaces that are supported by both lite and full protobuf.
CAFFE2_API bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto);
inline bool ReadProtoFromBinaryFile(const string filename, MessageLite* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

CAFFE2_API void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename);
inline void WriteProtoToBinaryFile(const MessageLite& proto,
                                   const string& filename) {
  return WriteProtoToBinaryFile(proto, filename.c_str());
}

#ifdef CAFFE2_USE_LITE_PROTO

namespace TextFormat {
inline bool ParseFromString(const string& spec, MessageLite* proto) {
  LOG(FATAL) << "If you are running lite version, you should not be "
             << "calling any text-format protobuffers.";
  return false;
}
} // namespace TextFormat


CAFFE2_API string ProtoDebugString(const MessageLite& proto);

CAFFE2_API bool ParseProtoFromLargeString(const string& str, MessageLite* proto);

// Text format MessageLite wrappers: these functions do nothing but just
// allowing things to compile. It will produce a runtime error if you are using
// MessageLite but still want text support.
inline bool ReadProtoFromTextFile(
    const char* /*filename*/,
    MessageLite* /*proto*/) {
  LOG(FATAL) << "If you are running lite version, you should not be "
                  << "calling any text-format protobuffers.";
  return false;  // Just to suppress compiler warning.
}
inline bool ReadProtoFromTextFile(const string filename, MessageLite* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void WriteProtoToTextFile(
    const MessageLite& /*proto*/,
    const char* /*filename*/) {
  LOG(FATAL) << "If you are running lite version, you should not be "
                  << "calling any text-format protobuffers.";
}
inline void WriteProtoToTextFile(const MessageLite& proto,
                                 const string& filename) {
  return WriteProtoToTextFile(proto, filename.c_str());
}

inline bool ReadProtoFromFile(const char* filename, MessageLite* proto) {
  return (ReadProtoFromBinaryFile(filename, proto) ||
          ReadProtoFromTextFile(filename, proto));
}

inline bool ReadProtoFromFile(const string& filename, MessageLite* proto) {
  return ReadProtoFromFile(filename.c_str(), proto);
}

#else  // CAFFE2_USE_LITE_PROTO

using ::google::protobuf::Message;

namespace TextFormat {
CAFFE2_API bool ParseFromString(const string& spec, Message* proto);
} // namespace TextFormat

CAFFE2_API string ProtoDebugString(const Message& proto);

CAFFE2_API bool ParseProtoFromLargeString(const string& str, Message* proto);

CAFFE2_API bool ReadProtoFromTextFile(const char* filename, Message* proto);
inline bool ReadProtoFromTextFile(const string filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

CAFFE2_API void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  return WriteProtoToTextFile(proto, filename.c_str());
}

// Read Proto from a file, letting the code figure out if it is text or binary.
inline bool ReadProtoFromFile(const char* filename, Message* proto) {
  return (ReadProtoFromBinaryFile(filename, proto) ||
          ReadProtoFromTextFile(filename, proto));
}

inline bool ReadProtoFromFile(const string& filename, Message* proto) {
  return ReadProtoFromFile(filename.c_str(), proto);
}

#endif  // CAFFE2_USE_LITE_PROTO

template <
    class IterableInputs = std::initializer_list<string>,
    class IterableOutputs = std::initializer_list<string>,
    class IterableArgs = std::initializer_list<Argument>>
OperatorDef CreateOperatorDef(
    const string& type,
    const string& name,
    const IterableInputs& inputs,
    const IterableOutputs& outputs,
    const IterableArgs& args,
    const DeviceOption& device_option = DeviceOption(),
    const string& engine = "") {
  OperatorDef def;
  def.set_type(type);
  def.set_name(name);
  for (const string& in : inputs) {
    def.add_input(in);
  }
  for (const string& out : outputs) {
    def.add_output(out);
  }
  for (const Argument& arg : args) {
    def.add_arg()->CopyFrom(arg);
  }
  if (device_option.has_device_type()) {
    def.mutable_device_option()->CopyFrom(device_option);
  }
  if (engine.size()) {
    def.set_engine(engine);
  }
  return def;
}

// A simplified version compared to the full CreateOperator, if you do not need
// to specify args.
template <
    class IterableInputs = std::initializer_list<string>,
    class IterableOutputs = std::initializer_list<string>>
inline OperatorDef CreateOperatorDef(
    const string& type,
    const string& name,
    const IterableInputs& inputs,
    const IterableOutputs& outputs,
    const DeviceOption& device_option = DeviceOption(),
    const string& engine = "") {
  return CreateOperatorDef(
      type,
      name,
      inputs,
      outputs,
      std::vector<Argument>(),
      device_option,
      engine);
}

CAFFE2_API bool HasOutput(const OperatorDef& op, const std::string& output);
CAFFE2_API bool HasInput(const OperatorDef& op, const std::string& input);

/**
 * @brief A helper class to index into arguments.
 *
 * This helper helps us to more easily index into a set of arguments
 * that are present in the operator. To save memory, the argument helper
 * does not copy the operator def, so one would need to make sure that the
 * lifetime of the OperatorDef object outlives that of the ArgumentHelper.
 */
class C10_EXPORT ArgumentHelper {
 public:
  template <typename Def>
  static bool HasArgument(const Def& def, const string& name) {
    return ArgumentHelper(def).HasArgument(name);
  }

  template <typename Def, typename T>
  static T GetSingleArgument(
      const Def& def,
      const string& name,
      const T& default_value) {
    return ArgumentHelper(def).GetSingleArgument<T>(name, default_value);
  }

  template <typename Def, typename T>
  static bool HasSingleArgumentOfType(const Def& def, const string& name) {
    return ArgumentHelper(def).HasSingleArgumentOfType<T>(name);
  }

  template <typename Def, typename T>
  static vector<T> GetRepeatedArgument(
      const Def& def,
      const string& name,
      const std::vector<T>& default_value = std::vector<T>()) {
    return ArgumentHelper(def).GetRepeatedArgument<T>(name, default_value);
  }

  template <typename Def, typename MessageType>
  static MessageType GetMessageArgument(const Def& def, const string& name) {
    return ArgumentHelper(def).GetMessageArgument<MessageType>(name);
  }

  template <typename Def, typename MessageType>
  static vector<MessageType> GetRepeatedMessageArgument(
      const Def& def,
      const string& name) {
    return ArgumentHelper(def).GetRepeatedMessageArgument<MessageType>(name);
  }

  template <typename Def>
  static bool RemoveArgument(Def& def, int index) {
    if (index >= def.arg_size()) {
      return false;
    }
    if (index < def.arg_size() - 1) {
      def.mutable_arg()->SwapElements(index, def.arg_size() - 1);
    }
    def.mutable_arg()->RemoveLast();
    return true;
  }

  explicit ArgumentHelper(const OperatorDef& def);
  explicit ArgumentHelper(const NetDef& netdef);
  bool HasArgument(const string& name) const;

  template <typename T>
  T GetSingleArgument(const string& name, const T& default_value) const;
  template <typename T>
  bool HasSingleArgumentOfType(const string& name) const;
  template <typename T>
  vector<T> GetRepeatedArgument(
      const string& name,
      const std::vector<T>& default_value = std::vector<T>()) const;

  template <typename MessageType>
  MessageType GetMessageArgument(const string& name) const {
    CAFFE_ENFORCE(arg_map_.count(name), "Cannot find parameter named ", name);
    MessageType message;
    if (arg_map_.at(name).has_s()) {
      CAFFE_ENFORCE(
          message.ParseFromString(arg_map_.at(name).s()),
          "Failed to parse content from the string");
    } else {
      VLOG(1) << "Return empty message for parameter " << name;
    }
    return message;
  }

  template <typename MessageType>
  vector<MessageType> GetRepeatedMessageArgument(const string& name) const {
    CAFFE_ENFORCE(arg_map_.count(name), "Cannot find parameter named ", name);
    vector<MessageType> messages(arg_map_.at(name).strings_size());
    for (int i = 0; i < messages.size(); ++i) {
      CAFFE_ENFORCE(
          messages[i].ParseFromString(arg_map_.at(name).strings(i)),
          "Failed to parse content from the string");
    }
    return messages;
  }

 private:
  CaffeMap<string, Argument> arg_map_;
};

// **** Arguments Utils *****

// Helper methods to get an argument from OperatorDef or NetDef given argument
// name. Throws if argument does not exist.
CAFFE2_API const Argument& GetArgument(const OperatorDef& def, const string& name);
CAFFE2_API const Argument& GetArgument(const NetDef& def, const string& name);
// Helper methods to get an argument from OperatorDef or NetDef given argument
// name. Returns nullptr if argument does not exist.
CAFFE2_API const Argument* GetArgumentPtr(const OperatorDef& def, const string& name);
CAFFE2_API const Argument* GetArgumentPtr(const NetDef& def, const string& name);

// Helper methods to query a boolean argument flag from OperatorDef or NetDef
// given argument name. If argument does not exist, return default value.
// Throws if argument exists but the type is not boolean.
CAFFE2_API bool GetFlagArgument(
    const OperatorDef& def,
    const string& name,
    bool default_value = false);
CAFFE2_API bool GetFlagArgument(
    const NetDef& def,
    const string& name,
    bool default_value = false);

CAFFE2_API Argument* GetMutableArgument(
    const string& name,
    const bool create_if_missing,
    OperatorDef* def);
CAFFE2_API Argument* GetMutableArgument(
    const string& name,
    const bool create_if_missing,
    NetDef* def);

template <typename T>
CAFFE2_API Argument MakeArgument(const string& name, const T& value);

template <typename T, typename Def>
inline void AddArgument(const string& name, const T& value, Def* def) {
  GetMutableArgument(name, true, def)->CopyFrom(MakeArgument(name, value));
}
// **** End Arguments Utils *****

bool inline operator==(const DeviceOption& dl, const DeviceOption& dr) {
  return IsSameDevice(dl, dr);
}

// Given a net, modify the external inputs/outputs if necessary so that
// the following conditions are met
// - No duplicate external inputs
// - No duplicate external outputs
// - Going through list of ops in order, all op inputs must be outputs
// from other ops, or registered as external inputs.
// - All external outputs must be outputs of some operators.
CAFFE2_API void cleanupExternalInputsAndOutputs(NetDef* net);

} // namespace caffe2

namespace std {
template <>
struct hash<caffe2::DeviceOption> {
  typedef caffe2::DeviceOption argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type const& device_option) const {
    std::string serialized;
    CAFFE_ENFORCE(device_option.SerializeToString(&serialized));
    return std::hash<std::string>{}(serialized);
  }
};
} // namespace std

#endif // CAFFE2_UTILS_PROTO_UTILS_H_
