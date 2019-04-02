#ifndef CAFFE2_UTILS_PROTO_UTILS_H_
#define CAFFE2_UTILS_PROTO_UTILS_H_

#include "caffe2/proto/caffe2.pb.h"
#include "google/protobuf/message.h"
#include "glog/logging.h"

namespace caffe2 {

using std::string;
using ::google::protobuf::Message;
using ::google::protobuf::MessageLite;
using std::string;

bool ReadProtoFromTextFile(const char* filename, Message* proto);
inline bool ReadProtoFromTextFile(const string filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  return WriteProtoToTextFile(proto, filename.c_str());
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

bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto);
inline bool ReadProtoFromBinaryFile(const string filename, MessageLite* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename);
inline void WriteProtoToBinaryFile(const MessageLite& proto,
                                   const string& filename) {
  return WriteProtoToBinaryFile(proto, filename.c_str());
}

// Read Proto from a file, letting the code figure out if it is text or binary.
inline bool ReadProtoFromFile(const char* filename, Message* proto) {
  return (ReadProtoFromBinaryFile(filename, proto) ||
          ReadProtoFromTextFile(filename, proto));
}
inline bool ReadProtoFromFile(const string& filename, Message* proto) {
  return ReadProtoFromFile(filename.c_str(), proto);
}

inline bool ReadProtoFromFile(const char* filename, MessageLite* proto) {
  return (ReadProtoFromBinaryFile(filename, proto) ||
          ReadProtoFromTextFile(filename, proto));
}
inline bool ReadProtoFromFile(const string& filename, MessageLite* proto) {
  return ReadProtoFromFile(filename.c_str(), proto);
}

// A coarse support for the Any message in proto3. I am a bit afraid of going
// directly to proto3 yet, so let's do this first...
class Any {
  template <typename MessageType>
  static MessageType Parse(const Argument& arg) {
    CHECK_EQ(arg.strings_size(), 1)
        << "An Any object should parse from a single string.";
    MessageType message;
    CHECK(message.ParseFromString(arg.strings(0)))
        << "Faild to parse from the string";
    return message;
  }
};

}  // namespace caffe2

#endif  // CAFFE2_UTILS_PROTO_UTILS_H_
