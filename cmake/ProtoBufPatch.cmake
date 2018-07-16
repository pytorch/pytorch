# CMake file to replace the string contents in ONNX, Caffe, and Caffe2 proto.
# Usage example:
#   cmake -DFILENAME=caffe2.pb.h -P ProtoBufPatch.cmake

file(READ ${FILENAME} content)

string(
  REPLACE
  "::google::protobuf::internal::fixed_address_empty_string"
  "fixed_address_empty_string"
  content
  "${content}")

string(
  REPLACE
  "::google::protobuf::internal::GetEmptyStringAlreadyInited"
  "GetEmptyStringAlreadyInited"
  content
  "${content}")

string(
  REPLACE
  "::google::protobuf::internal::GetEmptyString"
  "GetEmptyString"
  content
  "${content}")

foreach(ns ${NAMESPACES})
  string(
    REPLACE
    "namespace ${ns} {"
    "namespace ${ns} { ::google::protobuf::internal::ExplicitlyConstructed< ::std::string>& fixed_address_empty_string(); const ::std::string& GetEmptyStringAlreadyInited(); const ::std::string& GetEmptyString();"
    content
    "${content}")
endforeach()

file(WRITE ${FILENAME} "${content}")
