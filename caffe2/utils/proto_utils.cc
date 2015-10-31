#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <fstream>

#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

namespace caffe2 {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;
using google::protobuf::MessageLite;

using std::fstream;
using std::ios;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CAFFE_CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CAFFE_CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto) {
  int fd = open(filename, O_RDONLY);
  CAFFE_CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  // A hack to manually allow using very large protocol buffers.
  coded_input->SetTotalBytesLimit(1073741824, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CAFFE_CHECK_NE(fd, -1) << "File cannot be created: " << filename
                   << " error number: " << errno;
  ZeroCopyOutputStream* raw_output = new FileOutputStream(fd);
  CodedOutputStream* coded_output = new CodedOutputStream(raw_output);
  CAFFE_CHECK(proto.SerializeToCodedStream(coded_output));
  delete coded_output;
  delete raw_output;
  close(fd);
}

}  // namespace caffe2
