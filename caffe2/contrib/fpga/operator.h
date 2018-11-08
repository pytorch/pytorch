#ifndef CAFFE2_FB_OPENCL_OPERATOR_H_
#define CAFFE2_FB_OPENCL_OPERATOR_H_

#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

C10_DECLARE_REGISTRY(
    OpenCLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_OPENCL_OPERATOR_CREATOR(key, ...) \
  C10_REGISTER_CREATOR(OpenCLOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_OPENCL_OPERATOR(name, ...)                           \
  extern void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();         \
  static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_OPENCL##name() { \
    CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                   \
  }                                                                   \
  C10_REGISTER_CLASS(OpenCLOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_OPENCL_OPERATOR_STR(str_name, ...) \
  C10_REGISTER_TYPED_CLASS(OpenCLOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_OPENCL_OPERATOR_WITH_ENGINE(name, engine, ...) \
  C10_REGISTER_CLASS(                                           \
      OpenCLOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

} // namespace caffe2

#endif // CAFFE2_FB_OPENCL_OPERATOR_H_
