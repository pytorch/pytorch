# CMake file to replace the string contents in ONNX, Caffe, and Caffe2 proto.
# Usage example:
#   cmake -DFILENAME=caffe2.pb.h -P ProtoBufPatch.cmake

file(READ ${FILENAME} content)

string(
  REPLACE
  "::google::protobuf::internal::GetEmptyStringAlreadyInited"
  "GetEmptyStringAlreadyInited"
  content
  "${content}")

foreach(ns ${NAMESPACES})
  string(
    REPLACE
    "namespace ${ns} {"
    "namespace ${ns} { const ::std::string& GetEmptyStringAlreadyInited(); "
    content
    "${content}")
endforeach()

file(WRITE ${FILENAME} "${content}")
