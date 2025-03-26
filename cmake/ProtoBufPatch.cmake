# CMake file to replace the string contents in ONNX, Caffe, and Caffe2 proto.
# Usage example:
#   cmake -DFILENAME=caffe2.pb.h -DLOCAL_PROTOBUF=ON -P ProtoBufPatch.cmake

file(READ ${FILENAME} content)

# constexpr int TensorBoundShape_DimType_DimType_ARRAYSIZE = TensorBoundShape_DimType_DimType_MAX + 1;
# throws
# error: more than one operator "+" matches these operands:
#     built-in operator "arithmetic + arithmetic"
#     function "c10::operator+(int, c10::BFloat16)"
#     function "c10::operator+(c10::BFloat16, int)"
#     function "c10::operator+(int, c10::Half)"
#     function "c10::operator+(c10::Half, int)"
#   operand types are: const caffe2::ExternalDataProto_SourceType + int
string(
  REGEX REPLACE
  "constexpr ([^ ]+) ([^ ]+_ARRAYSIZE) = ([^ ]+_MAX) \\+ 1;"
  "constexpr \\1 \\2 = static_cast<\\1>(\\3) + 1;"
  content
  "${content}")

file(WRITE ${FILENAME} "${content}")
