# CMake file to replace the string contents in ONNX, Caffe, and Caffe2 proto.
# Usage example:
#   cmake -DFILENAME=caffe2.pb.h -P ProtoBufPatch.cmake

file(READ ${FILENAME} content)

# protobuf-3.6.0 pattern
string(
  REPLACE
  "::google::protobuf::internal::GetEmptyStringAlreadyInited"
  "GetEmptyStringAlreadyInited"
  content
  "${content}")

# protobuf-3.8.0+ pattern
string(
  REPLACE
  "::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited"
  "GetEmptyStringAlreadyInited"
  content
  "${content}")

string(
  REPLACE
  "PROTOBUF_CONSTEXPR"
  ""
  content
  "${content}")

# https://github.com/protocolbuffers/protobuf/commit/0400cca3236de1ca303af38bf81eab332d042b7c
# changes PROTOBUF_CONSTEXPR to constexpr, which breaks windows
# build.
string(
  REGEX REPLACE
  "static constexpr ([^ ]+) ([^ ]+) ="
  "static \\1 const \\2 ="
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

# The moving constructor is defined in the header file, which will cause
# a link error that claims that the vftable is not found. Luckily, we
# could move the definition into the source file to solve the problem.
list(LENGTH NAMESPACES ns_count)
if ("${FILENAME}" MATCHES ".pb.h" AND ns_count EQUAL 1)
  string(REPLACE ".pb.h" ".pb.cc" SOURCE_FILENAME ${FILENAME})
  file(READ ${SOURCE_FILENAME} content_cc_origin)

  string(REGEX MATCHALL "([a-zA-Z_]+)\\([a-zA-Z_]+&& from\\) noexcept[^}]*}" content_cc "${content}")
  string(REGEX REPLACE "};" "}\n" content_cc "${content_cc}")
  string(REGEX REPLACE "([a-zA-Z_]+)\\([a-zA-Z_]+&& from\\) noexcept" "  \\1::\\1(\\1&& from) noexcept" content_cc "${content_cc}")
  set(content_cc "${content_cc_origin}\nnamespace ${NAMESPACES} {\n#if LANG_CXX11\n${content_cc}\n#endif\n}")

  string(REGEX REPLACE "([a-zA-Z_]+)\\([a-zA-Z_]+&& from\\) noexcept([^}]*)}" "\\1(\\1&& from) noexcept;" content "${content}")

  file(WRITE ${SOURCE_FILENAME} "${content_cc}")
endif()

file(WRITE ${FILENAME} "${content}")
