#pragma once

#include <caffe2/utils/proto_utils.h>

namespace caffe2 {

namespace cast {

inline TensorProto_DataType GetCastDataType(const ArgumentHelper& helper, std::string arg) {
  TensorProto_DataType to;
  if (helper.HasSingleArgumentOfType<string>(arg)) {
    string s = helper.GetSingleArgument<string>(arg, "float");
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
#ifndef CAFFE2_USE_LITE_PROTO
    CAFFE_ENFORCE(TensorProto_DataType_Parse(s, &to), "Unknown 'to' argument: ", s);
#else

// Manually implement in the lite proto case.
#define X(t)                         \
  if (s == #t) {                     \
    return TensorProto_DataType_##t; \
  }

    X(FLOAT);
    X(INT32);
    X(BYTE);
    X(STRING);
    X(BOOL);
    X(UINT8);
    X(INT8);
    X(UINT16);
    X(INT16);
    X(INT64);
    X(FLOAT16);
    X(DOUBLE);
#undef X
    CAFFE_THROW("Unhandled type argument: ", s);

#endif
  } else {
    to = static_cast<TensorProto_DataType>(
        helper.GetSingleArgument<int>(arg, TensorProto_DataType_FLOAT));
  }
  return to;
}

};  // namespace cast

};  // namespace caffe2
