# CMake file to replace the string contents in Caffe and Caffe2 proto.
# Usage example:
#   cmake -DFILENAME=caffe2.pb.h -P ProtoBufPatch.cmake

file(READ ${FILENAME} content)
string(
  REPLACE
  "::google::protobuf::internal::GetEmptyStringAlreadyInited"
  "GetEmptyStringAlreadyInited"
  content
  "${content}")
string(REPLACE
	"namespace caffe2 {"
	"namespace caffe2 { const ::std::string& GetEmptyStringAlreadyInited(); "
	content
	"${content}")
string(REPLACE
	"namespace caffe {"
	"namespace caffe { const ::std::string& GetEmptyStringAlreadyInited(); "
	content
	"${content}")
file(WRITE ${FILENAME} "${content}")
