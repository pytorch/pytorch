#ifndef CAFFE2_UTILS_PROTO_UTILS_H_
#define CAFFE2_UTILS_PROTO_UTILS_H_

#include "google/protobuf/message_lite.h"
#ifndef CAFFE2_USE_LITE_PROTO
#include "google/protobuf/message.h"
#endif  // !CAFFE2_USE_LITE_PROTO

#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2.pb.h"


namespace caffe2 {

using std::string;
using ::google::protobuf::MessageLite;

// Common interfaces that reads file contents into a string.
bool ReadStringFromFile(const char* filename, string* str);
bool WriteStringToFile(const string& str, const char* filename);

// Common interfaces that are supported by both lite and full protobuf.
bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto);
inline bool ReadProtoFromBinaryFile(const string filename, MessageLite* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename);
inline void WriteProtoToBinaryFile(const MessageLite& proto,
                                   const string& filename) {
  return WriteProtoToBinaryFile(proto, filename.c_str());
}

#ifdef CAFFE2_USE_LITE_PROTO

inline string ProtoDebugString(const MessageLite& proto) {
  return "(cannot show debug string for MessageLite)";
}

// Text format MessageLite wrappers: these functions do nothing but just
// allowing things to compile. It will produce a runtime error if you are using
// MessageLite but still want text support.
inline bool ReadProtoFromTextFile(const char* filename, MessageLite* proto) {
  LOG(FATAL) << "If you are running lite version, you should not be "
                  << "calling any text-format protobuffers.";
  return false;  // Just to suppress compiler warning.
}
inline bool ReadProtoFromTextFile(const string filename, MessageLite* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void WriteProtoToTextFile(const MessageLite& proto,
                                 const char* filename) {
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

inline string ProtoDebugString(const Message& proto) {
  return proto.DebugString();
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);
inline bool ReadProtoFromTextFile(const string filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
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


template <class IterableInputs, class IterableOutputs, class IterableArgs>
OperatorDef CreateOperatorDef(
    const string& type, const string& name, const IterableInputs& inputs,
    const IterableOutputs& outputs, const IterableArgs& args,
    const DeviceOption& device_option, const string& engine) {
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
// to specify device option or engine.
template <class IterableInputs, class IterableOutputs, class IterableArgs>
inline OperatorDef CreateOperatorDef(
    const string& type, const string& name, const IterableInputs& inputs,
    const IterableOutputs& outputs, const IterableArgs& args) {
  return CreateOperatorDef(
      type, name, inputs, outputs, args, DeviceOption(), "");
}

// A simplified version compared to the full CreateOperator, if you do not need
// to specify device option or engine or args.
template <class IterableInputs, class IterableOutputs>
inline OperatorDef CreateOperatorDef(
    const string& type, const string& name, const IterableInputs& inputs,
    const IterableOutputs& outputs) {
  return CreateOperatorDef(type, name, inputs, outputs,
                           std::vector<Argument>(), DeviceOption(), "");
}

inline bool HasArgument(const OperatorDef& def, const string& name) {
  for (const Argument& arg : def.arg()) {
    if (arg.name() == name) {
      return true;
    }
  }
  return false;
}

const Argument& GetArgument(const OperatorDef& def, const string& name);

Argument* GetMutableArgument(
    const string& name, const bool create_if_missing, OperatorDef* def);

template <typename T>
Argument MakeArgument(const string& name, const T& value);

template <typename T>
void AddArgument(const string& name, const T& value, OperatorDef* def);

}  // namespace caffe2

#endif  // CAFFE2_UTILS_PROTO_UTILS_H_
