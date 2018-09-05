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

string(
  REPLACE
  "constexpr"
  "const"
  content
  "${content}")

foreach(ns ${NAMESPACES})
  # Insert "const ::std::string& GetEmptyStringAlreadyInited();" within
  # the namespace and make sure we only do it once in the file. Unfortunately
  # using string(REPLACE ...) doesn't work because it will replace at all
  # locations and there might be multiple declarations of the namespace
  # depending on how the proto is structured.
  set(search "namespace ${ns} {")
  string(LENGTH "${search}" search_len)
  string(FIND "${content}" "${search}" pos)
  if (${pos} GREATER -1)
    math(EXPR pos "${pos}+${search_len}")
    string(SUBSTRING "${content}" 0 ${pos} content_pre)
    string(SUBSTRING "${content}" ${pos} -1 content_post)
    string(
      CONCAT
      content
      "${content_pre}"
      " const ::std::string& GetEmptyStringAlreadyInited(); "
      "${content_post}")
  endif()
endforeach()

file(WRITE ${FILENAME} "${content}")
